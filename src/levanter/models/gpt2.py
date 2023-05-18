from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Optional, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.compat.torch_serialization import (
    StateDict,
    StateDictSerializationMixin,
    apply_prefix,
    reshape_linear_layer,
    stack_state_dict,
    unstack_state_dict,
)


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
    def Pos(self) -> Axis:
        return Axis(name="position", size=self.seq_len)

    @property
    def KeyPos(self) -> Axis:
        return self.Pos.alias("key_position")

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
    def KeySize(self) -> Axis:
        return Axis(name="key_size", size=self.hidden_dim // self.num_heads)


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed

    def __init__(self, Embed: Axis, Mlp: Axis, activation_fn: Union[str, Callable], *, key, use_bias: bool = True):
        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = hnn.Linear(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias)
        self.c_proj = hnn.Linear(Out=Embed, In=Mlp, key=k_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        self.act = activation_fn  # type: ignore

    @named_call
    def __call__(self, hidden_states: NamedArray):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Gpt2Attention(StateDictSerializationMixin, eqx.Module):
    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    c_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    dropout: hnn.Dropout

    config: Gpt2Config = eqx.static_field()

    def __init__(self, config: Gpt2Config, *, key):
        self.config = config
        Qkv = Axis("qkv", size=3)
        use_bias = config.use_bias

        k_c, k_proj = jrandom.split(key, 2)
        self.c_attn = hnn.Linear(In=config.Embed, Out=(Qkv, config.Heads, config.KeySize), key=k_c, use_bias=use_bias)
        self.c_proj = hnn.Linear(In=(config.Heads, config.KeySize), Out=config.Embed, key=k_proj, use_bias=use_bias)
        self.dropout = hnn.Dropout(config.attn_pdrop)

    @named_call
    def __call__(self, x: NamedArray, mask: Optional[NamedArray], layer_idx, inference: bool = True, *, key):
        qkv_out = self.c_attn(x)
        q, k, v = qkv_out.unbind("qkv")

        # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
        scale = jax.lax.rsqrt(float(self.config.KeySize.size))
        if self.config.scale_attn_by_inverse_layer_idx:
            scale /= layer_idx + 1.0

        # do this first to help keep FP values small
        q = q * scale

        # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
        if self.config.upcast_attn:
            q = q.astype(jnp.float32)
            k = k.astype(jnp.float32)

        attn_scores = hax.dot("key_size", q, k)

        if mask is not None:
            attn_scores = attn_scores + (1.0 - mask) * -1e9

        attn_weights = hnn.softmax(attn_scores, axis="key_position").astype(x.dtype)
        attn_weights = self.dropout(attn_weights, key=key, inference=inference)

        attn_output = hax.dot("key_position", attn_weights, v)  # [heads, seq_len, head_dim]

        attn_output = self.c_proj(attn_output)
        return attn_output

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "Gpt2Attention":
        # our c_attn is [embed] -> [3, heads, head_dim] and hf's is the flattened [embed] -> [3 * heads * head_dim]
        # and our c_proj is [heads, head_dim] -> [embed] and hf's is the flattened [heads * head_dim] -> [embed]
        # so we need to reshape the one in the dict before forwarding to the linear
        # keep in mind that everything is vectorized in our implementation, so there's a leading num_layers dim

        es = cast(Axis, self.c_attn.In).size
        d = {}
        num_heads = self.config.Heads.size
        key_size = self.config.KeySize.size
        d.update(reshape_linear_layer(state_dict, apply_prefix(prefix, "c_attn"), (es,), (3, num_heads, key_size)))
        d.update(reshape_linear_layer(state_dict, apply_prefix(prefix, "c_proj"), (num_heads, key_size), (es,)))

        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # need to undo the reshape we did in from_state_dict
        # reminder that everything is vectorized
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix)

        es = cast(Axis, self.c_attn.In).size
        num_heads = self.config.Heads.size
        key_size = self.config.KeySize.size

        my_dict.update(
            reshape_linear_layer(my_dict, apply_prefix(prefix, "c_attn"), (es,), (3 * num_heads * key_size,))
        )
        my_dict.update(reshape_linear_layer(my_dict, apply_prefix(prefix, "c_proj"), (num_heads * key_size,), (es,)))

        state_dict.update(my_dict)
        return state_dict


class Gpt2Block(StateDictSerializationMixin, eqx.Module):
    ln_1: hnn.LayerNorm
    attn: Gpt2Attention
    ln_2: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    def __init__(self, config: Gpt2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        self.ln_1 = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(config, key=k_attn)
        self.ln_2 = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)
        self.mlp = Gpt2Mlp(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        self.resid_dropout = hnn.Dropout(pdrop=config.resid_pdrop)

    @named_call
    def __call__(self, hidden_states: NamedArray, mask: Optional[NamedArray], layer_idx, inference, *, key):
        k1, k2, k3 = haliax.jax_utils.maybe_rng_split(key, 3)

        attn_output = self.attn(self.ln_1(hidden_states), mask=mask, inference=inference, layer_idx=layer_idx, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = hidden_states + attn_output

        ff_output = self.mlp(self.ln_2(hidden_states))
        ff_output = self.resid_dropout(ff_output, key=k3, inference=inference)
        hidden_states = hidden_states + ff_output

        return hidden_states


class Gpt2Transformer(StateDictSerializationMixin, eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: Stacked[Gpt2Block]
    ln_f: hnn.LayerNorm

    @property
    def Layers(self) -> Axis:
        return self.config.Layers

    def __init__(self, config: Gpt2Config, *, key):
        super().__init__()
        self.config = config

        # vectorize the blocks
        self.blocks = Stacked(
            self.Layers,
            Gpt2Block,
            config,
            key=shaped_rng_split(key, config.num_layers),
            gradient_checkpointing=config.gradient_checkpointing,
        )
        self.ln_f = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)

    @named_call
    def __call__(self, hidden_states: NamedArray, attn_mask: Optional[NamedArray], *, inference, key) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        hidden_states = self.blocks.fold(hidden_states, attn_mask, hax.arange(self.Layers), inference, key=keys)
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"blocks": "h"}

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        # We use a vectorized set of blocks, meaning that we have 1 GptBlock,
        # whereas in hf we have numlayers GptBlocks. So we need to build one GptBlock from numlayers GptBlocks.
        # the individual blocks are named h.0.FOO, h.1.FOO, etc.
        # we want to vectorize them to h.FOO, h.FOO, etc.
        stacked = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "h"))
        out = super().from_state_dict(stacked, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        # this method needs to "devectorize" the blocks, so that we have a list of blocks h.0.FOO, h.1.FOO, etc.
        # first just do the normal thing with our own dict, which we'll post-process
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix)

        stacked_dict = unstack_state_dict(my_state_dict, apply_prefix(prefix, "h"))
        state_dict.update(stacked_dict)

        return state_dict


class Gpt2Embeddings(StateDictSerializationMixin, eqx.Module):
    token_embeddings: NamedArray
    position_embeddings: NamedArray
    dropout: hnn.Dropout

    Vocab: Axis = eqx.static_field()
    config: Gpt2Config = eqx.static_field()

    def __init__(self, Vocab: Axis, config: Gpt2Config, *, key):
        super().__init__()
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        self.Vocab = Vocab
        self.config = config

        self.token_embeddings = sharded_normal(k_wte, (Vocab, config.Embed)) * config.initializer_range
        self.position_embeddings = sharded_normal(k_wpe, (config.Pos, config.Embed)) * (config.initializer_range / 2)
        self.dropout = hnn.Dropout(pdrop=config.embed_pdrop)

    @named_call
    def embed(self, input_ids, inference, *, key):
        input_embeds = self.token_embeddings.take("vocab", input_ids)
        position_embeds = self.position_embeddings

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        return hidden_states

    def unembed(self, hidden_states: NamedArray):
        return hax.dot("embed", hidden_states, self.token_embeddings)

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"token_embeddings": "wte.weight", "position_embeddings": "wpe.weight"}


class Gpt2LMHeadModel(StateDictSerializationMixin, eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    def __init__(self, Vocab: Axis, config: Gpt2Config, *, key):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = Gpt2Transformer(config, key=k_t)
        self.embeddings = Gpt2Embeddings(Vocab, config, key=k_embeddings)

    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray], *, inference, key):
        if not inference and key is None:
            raise ValueError("key must be provided for training")

        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=inference, key=k_embed)
        hidden_states = self.transformer(hidden_states, attn_mask, inference=inference, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {"transformer": None, "embeddings": None}


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}
