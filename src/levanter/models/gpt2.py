import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from levanter.compat.torch_serialization import StateDict, TorchSerializationMixin, apply_prefix, reshape_linear_layer
from levanter.modeling_utils import ACT2FN


sharded_normal = hax.random.generate_sharded(hax.random.normal)


@dataclass(frozen=True)
class Gpt2Config:
    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # how much to scale the embedding dim for the mlp layer
    mlp_scale: int = 4

    initializer_range: float = 0.02
    # dropout doesn't really help so we 0 it out by default
    embed_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    # mistral tweaks:
    scale_attn_by_inverse_layer_idx: bool = False
    upcast_attn: bool = False

    gradient_checkpointing: bool = True  # better to just always use this
    gradient_checkpointing_block_size: int = 5

    use_bias: bool = True

    # Axes
    @property
    def SeqLen(self) -> Axis:
        return Axis(name="seqlen", size=self.seq_len)

    @property
    def KeySeqLen(self) -> Axis:
        return self.SeqLen.alias(f"key_{self.SeqLen.name}")

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    @property
    def Heads(self) -> Axis:
        return Axis(name="heads", size=self.num_heads)

    @property
    def Layers(self) -> Axis:
        return Axis(name="layers", size=self.num_layers)

    @property
    def Mlp(self) -> Axis:
        return Axis(name="mlp", size=self.hidden_dim * 4)

    @property
    def HeadDim(self) -> Axis:
        return Axis(name="head", size=self.hidden_dim // self.num_heads)


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed

    def __init__(
        self, Embed: Axis, Intermediate: Axis, activation_fn: Union[str, Callable], *, key, use_bias: bool = True
    ):
        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = hnn.Linear(Out=Intermediate, In=Embed, key=k_fc, use_bias=use_bias)
        self.c_proj = hnn.Linear(Out=Embed, In=Intermediate, key=k_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        self.act = activation_fn  # type: ignore

    @named_call
    def __call__(self, hidden_states: NamedArray):
        hidden_states = hax.auto_sharded(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Gpt2Attention(TorchSerializationMixin, eqx.Module):
    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    c_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    dropout: hnn.Dropout

    SeqLen: Axis = eqx.static_field()
    HeadDim: Axis = eqx.static_field()
    Heads: Axis = eqx.static_field()
    Qkv: Axis = eqx.static_field()
    KeySeqLen: Axis = eqx.static_field()

    # Mistral stability tweaks
    scale_by_inverse_layer_idx: bool = eqx.static_field()
    upcast: bool = eqx.static_field()

    def __init__(
        self,
        SeqLen: Axis,
        KeySeqLen: Axis,
        Embed: Axis,
        Heads: Axis,
        HeadDim: Axis,
        dropout_prob: float,
        scale_by_inverse_layer_idx: bool,
        upcast: bool,
        *,
        key,
        use_bias: bool = True,
    ):
        self.Heads = Heads
        self.HeadDim = HeadDim
        self.SeqLen = SeqLen
        self.Qkv = Axis("qkv", 3)
        self.KeySeqLen = KeySeqLen

        k_c, k_proj = jrandom.split(key, 2)
        self.c_attn = hnn.Linear(In=Embed, Out=(self.Qkv, self.Heads, self.HeadDim), key=k_c, use_bias=use_bias)
        self.c_proj = hnn.Linear(In=(self.Heads, self.HeadDim), Out=Embed, key=k_proj, use_bias=use_bias)
        self.dropout = hnn.Dropout(dropout_prob)

        self.scale_by_inverse_layer_idx = scale_by_inverse_layer_idx
        self.upcast = upcast

    @named_call
    def __call__(
        self, hidden_states: NamedArray, mask: Optional[NamedArray], layer_idx, inference: bool = True, *, key
    ):
        qkv_out = self.c_attn(hidden_states)
        q, k, v = qkv_out.unbind(self.Qkv)

        # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({self.SeqLen: self.KeySeqLen})
        v = v.rename({self.SeqLen: self.KeySeqLen})

        # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
        scale = jax.lax.rsqrt(float(self.HeadDim.size))
        if self.scale_by_inverse_layer_idx:
            scale /= layer_idx + 1.0

        # do this first to help keep FP values small
        q = q * scale

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.upcast:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)

        attn_scores = hax.dot(self.HeadDim, q, k)

        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask) * -1e9

        attn_weights = hnn.softmax(attn_scores, axis=self.KeySeqLen).astype(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights, key=key, inference=inference)

        attn_output = hax.dot(self.KeySeqLen, attn_weights, v)  # [heads, seq_len, head_dim]

        attn_output = self.c_proj(attn_output)
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
    ln_1: hnn.LayerNorm
    attn: Gpt2Attention
    ln_2: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    def __init__(self, config: Gpt2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        assert (
            config.Embed.size % config.num_heads == 0
        ), f"embed_dim={config.Embed} must be divisible by num_heads={config.num_heads}"

        self.ln_1 = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(
            SeqLen=config.SeqLen,
            KeySeqLen=config.KeySeqLen,
            Embed=config.Embed,
            Heads=config.Heads,
            HeadDim=config.HeadDim,
            dropout_prob=config.attn_pdrop,
            key=k_attn,
            scale_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            upcast=config.upcast_attn,
            use_bias=config.use_bias,
        )
        self.resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)
        self.ln_2 = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(
            Embed=config.Embed,
            Intermediate=config.Mlp,
            activation_fn=config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

    @named_call
    def __call__(self, hidden_states: NamedArray, mask: Optional[NamedArray], inference, layer_idx, *, key):
        k1, k2, k3 = haliax.jax_utils.maybe_rng_split(key, 3)

        hidden_states = hax.auto_sharded(hidden_states)
        attn_output = self.attn(self.ln_1(hidden_states), mask=mask, inference=inference, layer_idx=layer_idx, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = hidden_states + attn_output

        ff_output = self.mlp(self.ln_2(hidden_states))
        ff_output = self.resid_dropout(ff_output, key=k3, inference=inference)
        hidden_states = hidden_states + ff_output

        return hidden_states


class Gpt2Transformer(TorchSerializationMixin, eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: Gpt2Block
    ln_f: hnn.LayerNorm

    @property
    def Layers(self) -> Axis:
        return self.config.Layers

    def __init__(self, config: Gpt2Config, *, key):
        super().__init__()
        self.config = config

        # vectorize the blocks
        self.blocks = hax.vmap(Gpt2Block, self.Layers)(config, key=shaped_rng_split(key, config.num_layers))
        self.ln_f = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)

    @named_call
    def __call__(self, hidden_states: NamedArray, attn_mask: Optional[NamedArray], *, inference, key) -> NamedArray:
        def do_block(hidden_states, block, layer_idx, key):
            return block(hidden_states, attn_mask, inference=inference, layer_idx=layer_idx, key=key)

        if self.config.gradient_checkpointing:
            do_block = jax.checkpoint(do_block, prevent_cse=False)

        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        hidden_states = hax.fold(do_block, self.Layers)(  # type: ignore
            hidden_states, self.blocks, hax.arange(self.Layers), key=keys  # type: ignore
        )
        hidden_states = hax.auto_sharded(hidden_states)
        hidden_states = self.ln_f(hidden_states)

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
    # position_embeddings: NamedArray
    token_out_embeddings: Optional[NamedArray]
    dropout: hnn.Dropout

    # axes
    Vocab: Axis = eqx.static_field()
    SeqLen: Axis = eqx.static_field()
    Embed: Axis = eqx.static_field()

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
    ):
        super().__init__()
        k_wte, k_out = jrandom.split(key, 2) # k_wpe,

        self.Vocab = Vocab
        self.SeqLen = SeqLen
        self.Embed = Embed

        self.token_embeddings = sharded_normal(key=k_wte, shape=(Vocab, Embed)) * initializer_range
        # self.position_embeddings = sharded_normal(key=k_wpe, shape=(SeqLen, Embed)) * (initializer_range / 2)
        self.dropout = hnn.Dropout(pdrop=dropout_prob)

        if tie_word_embeddings:
            self.token_out_embeddings = None
        else:
            self.token_out_embeddings = sharded_normal(key=k_out, shape=(Vocab, Embed)) * initializer_range

    @named_call
    def embed(self, input_ids, inference, *, key):
        input_embeds = self.token_embeddings.take(self.Vocab, input_ids)
        #position_embeds = self.position_embeddings

        hidden_states = input_embeds #+ position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        return hidden_states

    def unembed(self, hidden_states: NamedArray):
        embeddings = self.token_out_embeddings or self.token_embeddings
        return hax.dot(self.Embed, hidden_states, embeddings)

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        assert self.token_out_embeddings is None
        return {"token_embeddings": "wte.weight"} # , "position_embeddings": "wpe.weight"}


class Gpt2LMHeadModel(TorchSerializationMixin, eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.embeddings.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def SeqLen(self) -> Axis:
        return self.embeddings.SeqLen

    def __init__(self, Vocab: Axis, config: Gpt2Config, *, key):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = Gpt2Transformer(config, key=k_t)
        self.embeddings = Gpt2Embeddings(
            Vocab=Vocab,
            Embed=config.Embed,
            SeqLen=config.SeqLen,
            initializer_range=config.initializer_range,
            tie_word_embeddings=True,
            dropout_prob=config.embed_pdrop,
            key=k_embeddings,
        )

    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray], *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, attn_mask, inference=inference, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits

    def _torch_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"transformer": None, "embeddings": None}
