import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, cast

import equinox as eqx
import equinox.nn as nn
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jmp
from equinox.custom_types import Array

import haliax as hax
import levanter.nn as pnn
from haliax import Axis, NamedArray
from haliax.nn.linear import Linear
from haliax.partitioning import logically_sharded
from levanter import jax_utils
from levanter.compat.torch_serialization import StateDict, TorchSerializationMixin, apply_prefix, reshape_linear_layer
from levanter.jax_utils import named_call
from levanter.modeling_utils import ACT2FN


@dataclass(frozen=True)
class Gpt2Config:
    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    initializer_range: float = 0.02
    embed_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    # mistral tweak:
    scale_attn_by_inverse_layer_idx: bool = False

    gradient_checkpointing: bool = False
    gradient_checkpointing_block_size: int = 5

    # Axes
    @property
    def SeqLen(self) -> Axis:
        return Axis(name="seqlen", size=self.seq_len)

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    @property
    def Heads(self) -> Axis:
        return Axis(name="heads", size=self.num_heads)

    @property
    def Layers(self) -> Axis:
        return Axis(name="layers", size=self.num_layers)


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: Linear
    c_proj: Linear

    def __init__(self, Embed: Axis, Intermediate: Axis, activation_fn, *, key, mp: jmp.Policy):
        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = Linear(Out=Intermediate, In=Embed, key=k_fc, mp=mp)
        self.c_proj = Linear(Out=Embed, In=Intermediate, key=k_proj, mp=mp)
        self.act = ACT2FN[activation_fn]  # type: ignore

    @named_call
    def __call__(self, hidden_states: NamedArray):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = jax.tree_util.tree_map(self.act, hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Gpt2Attention(TorchSerializationMixin, eqx.Module):
    c_attn: Linear
    c_proj: Linear
    dropout: pnn.Dropout

    causal: bool = eqx.static_field()
    SeqLen: Axis = eqx.static_field()
    HeadDim: Axis = eqx.static_field()
    Heads: Axis = eqx.static_field()
    Qkv: Axis = eqx.static_field()

    scale_by_inverse_layer_idx: bool = eqx.static_field()
    mp: jmp.Policy = eqx.static_field()

    def __init__(
        self,
        SeqLen: Axis,
        InDim: Axis,
        Heads: Axis,
        HeadDim: Axis,
        dropout_prob: float,
        scale_by_inverse_layer_idx: bool,
        *,
        key,
        mp: jmp.Policy,
        causal: bool = True,
    ):
        self.causal = causal
        self.Heads = Heads
        self.HeadDim = HeadDim
        self.SeqLen = SeqLen

        self.Qkv = Axis("qkv", 3)
        self.scale_by_inverse_layer_idx = scale_by_inverse_layer_idx
        self.mp = mp

        k_c, k_proj = jrandom.split(key, 2)

        self.c_attn = Linear(Out=(self.Qkv, self.Heads, self.HeadDim), In=InDim, key=k_c, mp=mp)
        self.c_proj = Linear(Out=InDim, In=(self.Heads, self.HeadDim), key=k_proj, mp=mp)
        self.dropout = pnn.Dropout(dropout_prob)

    # TODO: cross-attention
    @named_call
    def __call__(self, hidden_states: NamedArray, layer_idx, inference: bool = True, *, key):
        # hidden_states has shape [seq_len, embed_dim]
        rng_key = key

        qkv_out = logically_sharded(self.c_attn(hidden_states))  # [seq_len, 3, heads, head_dim]
        query, key, value = logically_sharded(qkv_out.unbind(self.Qkv))

        KeySeqLen = self.SeqLen.alias("KeySeqLen")  # haliax doesn't support unnamed axes or duplicate axes
        key = key.rename({self.SeqLen: KeySeqLen})
        value = value.rename({self.SeqLen: KeySeqLen})

        # mistral tweak
        scale = lax.rsqrt(float(self.HeadDim.size))
        if self.scale_by_inverse_layer_idx:
            scale /= layer_idx + 1.0

        # do this first to help keep FP values small
        query = query * scale

        attn_weights = hax.dot(self.HeadDim, query, key)
        attn_weights = hax.rearrange(attn_weights, (..., self.Heads, self.SeqLen, KeySeqLen))
        attn_axes = attn_weights.axes
        attn_weights = attn_weights.array

        if self.causal:
            # TODO(haliax) add tril to hax
            causal_mask = jnp.tril(jnp.ones((self.SeqLen.size, KeySeqLen.size), dtype=jnp.bool_))

            # TODO(haliax): add where ops to hax
            attn_weights = jnp.where(causal_mask, attn_weights, -1e9)

        attn_weights = jnn.softmax(attn_weights)  # heads, seqlen, seqlen
        attn_weights = self.mp.cast_to_compute(attn_weights)
        attn_weights = self.dropout(attn_weights, key=rng_key, inference=inference)
        attn_weights = NamedArray(attn_weights, attn_axes)

        attn_output = hax.dot(KeySeqLen, attn_weights, value)  # [heads, seq_len, head_dim]

        attn_output = self.c_proj(attn_output)

        assert attn_output.dtype == self.mp.compute_dtype

        return attn_output

    def from_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None) -> "Gpt2Attention":
        # our c_attn is [embed] -> [3, heads, head_dim] and torch's is the flattened [embed] -> [3 * heads * head_dim]
        # and our c_proj is [heads, head_dim] -> [embed] and torch's is the flattened [heads * head_dim] -> [embed]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        es = cast(Axis, self.c_attn.In).size
        d = {}
        d.update(
            reshape_linear_layer(
                torch_dict, apply_prefix(prefix, "c_attn"), (es,), (3, self.Heads.size, self.HeadDim.size)
            )
        )
        d.update(
            reshape_linear_layer(
                torch_dict, apply_prefix(prefix, "c_proj"), (self.Heads.size, self.HeadDim.size), (es,)
            )
        )

        return super().from_torch_dict(d, prefix)

    def update_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_torch_dict
        # reminder that everything is vectorized
        my_dict: StateDict = {}
        super().update_torch_dict(my_dict, prefix)

        es = cast(Axis, self.c_attn.In).size
        my_dict.update(
            reshape_linear_layer(
                my_dict, apply_prefix(prefix, "c_attn"), (es,), (3 * self.Heads.size * self.HeadDim.size,)
            )
        )
        my_dict.update(
            reshape_linear_layer(
                my_dict, apply_prefix(prefix, "c_proj"), (self.Heads.size * self.HeadDim.size,), (es,)
            )
        )

        torch_dict.update(my_dict)
        return torch_dict


class Gpt2Block(TorchSerializationMixin, eqx.Module):
    mp: jmp.Policy = eqx.static_field()
    ln_1: nn.LayerNorm
    attn: Gpt2Attention
    ln_2: nn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: pnn.Dropout

    SeqLen: Axis = eqx.static_field()
    Embed: Axis = eqx.static_field()

    def __init__(self, config: Gpt2Config, *, key, mp: jmp.Policy):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        self.Embed = config.Embed
        self.SeqLen = config.SeqLen
        Mlp = Axis("mlp", 4 * config.Embed.size)
        HeadDim = Axis("head", config.Embed.size // config.num_heads)

        assert (
            config.Embed.size % config.num_heads == 0
        ), f"embed_dim={config.Embed} must be divisible by num_heads={config.num_heads}"

        self.mp = mp

        self.ln_1 = nn.LayerNorm(config.Embed.size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(
            SeqLen=config.SeqLen,
            InDim=config.Embed,
            Heads=config.Heads,
            HeadDim=HeadDim,
            dropout_prob=config.attn_pdrop,
            key=k_attn,
            causal=True,
            scale_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            mp=mp,
        )
        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)
        self.ln_2 = nn.LayerNorm(config.Embed.size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(
            Embed=config.Embed,
            Intermediate=Mlp,
            activation_fn=config.activation_function,
            key=k_mlp,
            mp=mp,
        )

    @named_call
    def __call__(self, hidden_states: Array, inference, layer_idx, *, key):
        k1, k2, k3 = jax_utils.maybe_rng_split(key, 3)

        residual = hidden_states
        hidden_states = jax.vmap(self.ln_1)(hidden_states)
        hidden_states = self.mp.cast_to_compute(hidden_states)
        h = NamedArray(hidden_states, (self.SeqLen, self.Embed))
        attn_output = self.attn(h, inference=inference, layer_idx=layer_idx, key=k1)
        dout = self.resid_dropout(attn_output.array, key=k2, inference=inference)
        hidden_states = residual + dout

        residual = hidden_states
        hidden_states = jax.vmap(self.ln_2)(hidden_states)
        hidden_states = self.mp.cast_to_compute(hidden_states)
        h = NamedArray(hidden_states, (self.SeqLen, self.Embed))
        ff_output = self.mlp(h)
        dout = self.resid_dropout(ff_output.array, key=k3, inference=inference)
        hidden_states = residual + dout

        assert attn_output.dtype == self.mp.compute_dtype

        return hidden_states


class Gpt2Transformer(TorchSerializationMixin, eqx.Module):
    config: Gpt2Config = eqx.static_field()
    mp: jmp.Policy = eqx.static_field()
    blocks: Gpt2Block
    ln_f: nn.LayerNorm

    @property
    def Layers(self) -> Axis:
        return self.config.Layers

    def __init__(self, config: Gpt2Config, *, key, mp: jmp.Policy):
        super().__init__()
        self.config = config
        self.mp = mp

        self.blocks = hax.vmap(lambda key: Gpt2Block(config, key=key, mp=mp), self.Layers)(
            jax_utils.shaped_rng_split(key, config.num_layers),
        )
        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)

    @named_call
    def __call__(self, hidden_states: Array, inference=True, *, key) -> Array:
        def do_block(hidden_states, block_layer_idx_key):
            block, layer_idx = block_layer_idx_key[0], block_layer_idx_key[1]
            if len(block_layer_idx_key) > 2:
                block_key = block_layer_idx_key[2]
            else:
                block_key = (
                    None  # key is none when we are in inference mode, and there's no way to scan over None keys
                )
            return block(hidden_states, inference=inference, layer_idx=layer_idx, key=block_key)

        if self.config.gradient_checkpointing:
            do_block = jax.checkpoint(do_block)

        if key is None:
            hidden_states = hax.reduce(
                do_block, self.Layers, hidden_states, (self.blocks, jnp.arange(self.Layers.size))
            )
        else:
            keys = jax_utils.maybe_rng_split(key, self.Layers.size)
            hidden_states = hax.reduce(
                do_block, self.Layers, hidden_states, (self.blocks, jnp.arange(self.Layers.size), keys)
            )

        hidden_states = jax.vmap(self.ln_f)(hidden_states)
        hidden_states = self.mp.cast_to_compute(hidden_states)

        return hidden_states

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"blocks": "h"}

    def from_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None):
        import torch

        # this method is a bit of a pain because we use a vectorized set of blocks, meaning that we have 1 GptBlock,
        # whereas in torch we have numlayers GptBlocks. So we need to build one GptBlock from numlayers GptBlocks.
        # first we vectorize the keys for the torch dict
        # the individual blocks are named h.0.FOO, h.1.FOO, etc.
        # we want to vectorize them to h.FOO, h.FOO, etc.
        vectorized_dict: StateDict = {}

        tensors_to_vectorize: Dict[str, List[Optional[torch.Tensor]]] = {}
        prefix_to_vectorize = cast(str, apply_prefix(prefix, "h"))
        other_keys_prefix = cast(str, apply_prefix(prefix, ""))
        escaped = re.escape(prefix_to_vectorize)
        pattern = re.compile(rf"{escaped}\.(\d+)\.(.*)")
        for k, v in torch_dict.items():
            match = pattern.match(k)
            if match:
                block_idx = int(match.group(1))
                block_key = match.group(2)
                tensors = tensors_to_vectorize.setdefault(block_key, [None] * self.Layers.size)
                assert tensors[block_idx] is None, f"Duplicate key {k}"
                tensors[block_idx] = v
            elif k.startswith(other_keys_prefix):
                k = k[len(other_keys_prefix) :]
                vectorized_dict[k] = v

        # now we have to vectorize the tensors
        for k, tensors in tensors_to_vectorize.items():
            vectorized_dict[cast(str, apply_prefix("h", k))] = torch.stack(tensors, dim=0)

        # now we can just call the base class. No prefix is needed because we've stripped it
        out = super().from_torch_dict(vectorized_dict, prefix=None)
        return out

    def update_torch_dict(self, torch_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # this method is also a bit of a pain for the same reasons
        # first just do the normal thing with our own dict, which we'll post-process
        my_state_dict: StateDict = {}
        super().update_torch_dict(my_state_dict, prefix=None)

        # now go through and devectorize all the "h" keys
        for k, v in my_state_dict.items():
            if k.startswith("h."):
                # this is a vectorized key, we need to devectorize it
                unbound = v.unbind(dim=0)
                for i, v2 in enumerate(unbound):
                    torch_dict[cast(str, apply_prefix(prefix, f"h.{i}.{k[2:]}"))] = v2
            else:
                # other keys just copy over
                torch_dict[k] = v

        return torch_dict


class Gpt2Embeddings(TorchSerializationMixin, eqx.Module):
    token_embeddings: NamedArray
    position_embeddings: NamedArray
    token_out_embeddings: Optional[NamedArray]
    dropout: pnn.Dropout

    # axes
    Vocab: Axis = eqx.static_field()
    SeqLen: Axis = eqx.static_field()
    Embed: Axis = eqx.static_field()

    mp: jmp.Policy = eqx.static_field()

    def __init__(
        self,
        Embed: Axis,
        Vocab: Axis,
        SeqLen: Axis,
        initializer_range: float,
        tie_word_embeddings: bool,
        dropout_prob: float,
        *,
        key,
        mp: jmp.Policy,
    ):
        super().__init__()
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        self.Vocab = Vocab
        self.SeqLen = SeqLen
        self.Embed = Embed

        self.mp = mp

        self.token_embeddings = hax.random.normal(key=k_wte, shape=(Vocab, Embed)) * initializer_range
        self.position_embeddings = hax.random.normal(key=k_wpe, shape=(SeqLen, Embed)) * (initializer_range / 2)
        self.dropout = pnn.Dropout(p=dropout_prob)

        if tie_word_embeddings:
            self.token_out_embeddings = None
        else:
            self.token_out_embeddings = hax.random.normal(key=k_out, shape=(Vocab, Embed)) * initializer_range

    @named_call
    def embed(self, input_ids, inference, *, key):
        # TODO: select
        # input_embeds = self.token_embeddings.select(self.vocab, input_ids)
        # position_embeds = self.position_embeddings.select(self.seqlen, jnp.arange(input_ids.shape[-1], dtype="i4"))
        input_embeds = self.token_embeddings.array[input_ids]
        input_embeds = self.mp.cast_to_compute(input_embeds)

        position_embeds = self.position_embeddings.array[jnp.arange(input_ids.shape[-1], dtype="i4")]
        position_embeds = self.mp.cast_to_compute(position_embeds)

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        return hidden_states

    def unembed(self, hidden_states: Array):
        embeddings = self.token_out_embeddings or self.token_embeddings
        # return hax.dot(self.hidden, hidden_states, embeddings)
        return jnp.einsum("... l h, ... v h -> ... l v", hidden_states, embeddings.array)

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        assert self.token_out_embeddings is None
        return {"token_embeddings": "wte.weight", "position_embeddings": "wpe.weight"}


class Gpt2LMHeadModel(TorchSerializationMixin, eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.embeddings.Vocab.size

    def __init__(self, Vocab: Axis, config: Gpt2Config, *, key, mp: jmp.Policy):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = Gpt2Transformer(config, key=k_t, mp=mp)
        self.embeddings = Gpt2Embeddings(
            Vocab=Vocab,
            Embed=config.Embed,
            SeqLen=config.SeqLen,
            initializer_range=config.initializer_range,
            tie_word_embeddings=True,
            dropout_prob=config.embed_pdrop,
            key=k_embeddings,
            mp=mp,
        )

    def __call__(self, input_ids, *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, inference=inference, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"transformer": None, "embeddings": None}
