from functools import partial
from typing import Optional, List, Callable

import equinox as eqx
import equinox.nn as nn
import psithuros.nn as pnn
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange
from equinox.custom_types import Array
from transformers import GPT2Config

from psithuros.modeling_utils import ACT2FN


class Gpt2Conv1D(eqx.Module):
    kernel: Array
    bias: Optional[Array]

    def __init__(self, *, in_features: int, out_features: int, key):
        self.kernel = jrandom.normal(key, [out_features, in_features]) * 0.02
        self.bias = jnp.zeros(out_features)

    @jax.jit
    def __call__(self, inputs):
        kernel = jnp.transpose(self.kernel)
        return inputs @ kernel + self.bias


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: Gpt2Conv1D
    c_proj: Gpt2Conv1D
    dropout: pnn.Dropout

    def __init__(self, config, intermediate_size, *, key):
        embed_dim = config.hidden_size

        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = Gpt2Conv1D(out_features=intermediate_size, in_features=embed_dim, key=k_fc)
        self.c_proj = Gpt2Conv1D(out_features=embed_dim, in_features=intermediate_size, key=k_proj)
        self.act = ACT2FN[config.activation_function]
        self.dropout = pnn.Dropout(p=config.resid_pdrop)

    @eqx.filter_jit
    def __call__(self, hidden_states, *, inference: bool, key=None):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)
        return hidden_states


class Gpt2Attention(eqx.Module):
    causal: bool = eqx.static_field()
    embed_dim: int = eqx.static_field()
    num_heads: int = eqx.static_field()

    c_attn: Gpt2Conv1D
    c_proj: Gpt2Conv1D
    resid_dropout: pnn.Dropout

    causal_mask: Optional[Array]

    @property
    def head_dim(self):
        return self.embed_dim // self.num_heads

    def __init__(self, config: GPT2Config, *, key, causal: bool = True):
        self.causal = causal
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head

        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"

        k_c, k_q, k_proj = jrandom.split(key, 3)

        self.c_attn = Gpt2Conv1D(out_features=3 * self.embed_dim, in_features=self.embed_dim, key=k_c)
        self.c_proj = Gpt2Conv1D(out_features=self.embed_dim, in_features=self.embed_dim, key=k_proj)

        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)

        if self.causal:
            self.causal_mask = jnp.tril(jnp.ones((config.n_positions, config.n_positions)))
        else:
            self.causal_mask = None

    # TODO: cross-attention
    # TODO: reorder_and_upcast_attn
    # TODO: scale_attn_by_inverse_layer_idx
    @eqx.filter_jit
    def __call__(self, hidden_states, inference: bool = True, *, key):
        # hidden_states has shape [seq_len, embed_dim]
        rng_key = key

        qkv_out = self.c_attn(hidden_states)  # [seq_len, 3 * embed_dim]
        query, key, value = jnp.split(qkv_out, 3, axis=-1)  # [seq_len, embed_dim]

        query = self._split_heads(query)  # [num_heads, seq_len, head_dim]
        key = self._split_heads(key)
        value = self._split_heads(value)

        # must use negative indexing to please the pmap gods
        query_length, key_length = query.shape[-2], key.shape[-2]

        if self.causal:
            attention_mask = self.causal_mask[:query_length, :key_length]
        else:
            attention_mask = None

        w = jnp.einsum('... n d, ... m d -> ... n m', query, key)  # [heads, seq_len, seq_len]
        w = w * lax.rsqrt(float(value.shape[-1]))

        if attention_mask is not None:
            mask = jnp.broadcast_to(attention_mask, w.shape)
            w = jnp.where(mask > 0, w, -1E9)

        w = jnn.softmax(w)

        attn_output = jnp.einsum('... n m, ... m d -> ... n d', w, value)  # [heads, seq_len, head_dim]

        attn_output = self._merge_heads(attn_output)  # [seq_len, embed_dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, key=rng_key, inference=inference)

        return attn_output

    def _split_heads(self, hidden_states: Array["seq_len", "embed_dim"]) -> Array["num_heads", "seq_len", "head_dim"]:
        return rearrange(hidden_states, '... n (h d) -> ... h n d', h=self.num_heads)

    @staticmethod
    def _merge_heads(hidden_states: Array["num_heads", "seq_len", "head_dim"]) -> Array["seq_len", "embed_dim"]:
        return rearrange(hidden_states, '... h n d -> ... n (h d)')


class Gpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: Gpt2Attention
    ln_2: nn.LayerNorm
    mlp: Gpt2Mlp

    def __init__(self, config, *, key):
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(config, key=k_attn)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(config, inner_dim, key=k_mlp)

    @eqx.filter_jit
    def __call__(self, hidden_states, inference=True, *, key):
        k1, k2 = jrandom.split(key, 2) if key is not None else (None, None)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, inference=inference, key=k1)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states, inference=inference, key=k2)

        hidden_states = feed_forward_hidden_states + residual

        return hidden_states


class Gpt2Model(eqx.Module):
    config: GPT2Config = eqx.static_field()
    wte: jnp.ndarray
    wpe: jnp.ndarray
    dropout: pnn.Dropout
    blocks: List[Gpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: GPT2Config, *, key):
        super().__init__()
        self.config = config

        k_wte, k_wpe, k_blocks = jrandom.split(key, 3)
        embed_dim = config.n_embd

        self.wte = jrandom.normal(key=k_wte, shape=(config.vocab_size, embed_dim)) * config.initializer_range
        self.wpe = jrandom.normal(key=k_wpe,
                                  shape=(config.max_position_embeddings, embed_dim)) * config.initializer_range / 2

        self.dropout = pnn.Dropout(p=config.embd_pdrop)
        self.blocks = [
            Gpt2Block(config, key=k) for i, k in enumerate(jrandom.split(k_blocks, config.n_layer))
        ]
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)

    @eqx.filter_jit
    def __call__(self, input_ids: Array["seq_len"], inference=True, *, key):
        input_embeds = self.wte[input_ids]
        indices = jnp.arange(input_ids.shape[-1], dtype="i4")
        position_embeds = self.wpe[indices]

        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        keys = jrandom.split(key, len(self.blocks)) if key is not None else [None] * len(self.blocks)

        for block, k_block in zip(self.blocks, keys):
            hidden_states = block(hidden_states, inference=inference, key=k_block)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class Gpt2LMHeadModel(eqx.Module):
    transformer: Gpt2Model
    lm_head: jnp.ndarray

    @property
    def config(self):
        return self.transformer.config

    def __init__(self, config, *, key):
        k_t, k_lm_head = jrandom.split(key, 2)
        self.transformer = Gpt2Model(config, key=k_t)
        if config.tie_word_embeddings:
            self.lm_head = self.transformer.wte
        else:
            self.lm_head = jrandom.normal(k_lm_head,
                                          (config.vocab_size, config.hidden_size)) * config.initializer_range

    @eqx.filter_jit
    def __call__(self, input_ids: Array["seq_len"], inference=True, *, key):
        hidden_states = self.transformer(input_ids, inference=inference, key=key)
        lm_logits = hidden_states @ jnp.transpose(self.lm_head)

        return lm_logits
