from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, LmWithHfSerializationMixin
from levanter.compat.torch_serialization import StateDictSerializationMixin
from levanter.models.lm_model import LmConfig


@LmConfig.register_subclass("llama")
@dataclass(frozen=True)
class LlamaConfig(HFCompatConfig):
    """Config for LlamaModel

    Args:
        vocab_size (int, optional): vocabulary size of the Llama model. Defaults to 32000.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of key/value heads needed for Grouped Query Attention. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        max_position_embeddings (int, optional): maximum length of the position embedding. Defaults to 2048.
    """

    vocab_size: int = 32000
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    activation_function: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    use_bias: bool = True


class LlamaMlp(eqx.Module):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: hnn.Linear  # projection from Embed to Intermediate
    up_proj: hnn.Linear  # projection from Embed to Intermediate
    down_proj: hnn.Linear  # projection from Intermediate to Embed
    act: Callable = eqx.static_field()

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn, *, key, use_bias: bool = False) -> "LlamaMlp":
        k_fc, k_up_proj, k_down_proj = eqx.split_key(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn  # type: ignore

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        hidden_states = self.gate_proj(x)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x)
        outputs = self.down_proj(hidden_states)
        return outputs


class LlamaAttention(StateDictSerializationMixin, eqx.Module):
    config: LlamaConfig = eqx.static_field()
    q_proj: hnn.Linear  # projection from Embed to query
    k_proj: hnn.Linear  # projection from Embed to key
    v_proj: hnn.Linear  # projection from Embed to value
    o_proj: hnn.Linear  # projection from Heads to output
    # rotary_emb  # rotary embedding

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaAttention":
        use_bias = config.use_bias
        Embed = config.Embed

        k_q, k_k, k_v, k_o = eqx.split_key(key, 4)
        q_proj = hnn.Linear.init(In=Embed, Out=config.Heads * config.HeadDim, key=k_q, use_bias=use_bias)
        k_proj = hnn.Linear.init(In=Embed, Out=config.KVHeads * config.HeadDim, key=k_k, use_bias=use_bias)
        v_proj = hnn.Linear.init(In=Embed, Out=config.KVHeads * config.HeadDim, key=k_v, use_bias=use_bias)
        o_proj = hnn.Linear.init(In=config.Heads * config.HeadDim, Out=Embed, key=k_o, use_bias=use_bias)
        # rotary_emb = _get_rotary_emb(config)
        return LlamaAttention(config, q_proj, k_proj, v_proj, o_proj)

    named_call

    def __call__(self, x: NamedArray, mask: Optional[NamedArray], layer_idx, inference: bool = True, *, key):
        k_q, k_k, k_v = eqx.split_key(key, 3)


class LlamaBlock(StateDictSerializationMixin, eqx.Module):
    pass


class LlamaTransformer(StateDictSerializationMixin, eqx.Module):
    pass


class LlamaEmbeddings(StateDictSerializationMixin, eqx.Module):
    pass


class LlamaLMHeadModel(eqx.Module):
    pass


class LlamaRotaryEmbedding(eqx.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000
    inv_freq: jnp.ndarray = eqx.static_field()
    cos_cached: jnp.ndarray = eqx.static_field()
    sin_cached: jnp.ndarray = eqx.static_field()
    max_seq_len_cached: int = eqx.static_field()

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2) / self.dim))
        self.cos_cached, self.sin_cached = self._set_cos_sin_cache(seq_len=self.max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = jnp.arange(self.max_seq_len_cached)

        # Evaluates the Einstein summation convention on the operands.
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper but following HF implementation
        # It uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos_cached = jnp.cos(emb)[None, None, :, :]
        sin_cached = jnp.sin(emb)[None, None, :, :]

        return cos_cached, sin_cached

    def __call__(self, x, seq_len: int):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.cos_cached, self.sin_cached = self._set_cos_sin_cache(seq_len=seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...],
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling"""

    scaling_factor: float = 1.0

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len):
        """The main difference is that the scaling factor is applied to the time axis"""
        self.max_seq_len_cached = seq_len
        t = jnp.arange(self.max_seq_len_cached) / self.scaling_factor

        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos_cached = jnp.cos(emb)[None, None, :, :]
        sin_cached = jnp.sin(emb)[None, None, :, :]

        return cos_cached, sin_cached


def _get_rotary_emb(config: LlamaConfig):
    return None
