from functools import partial
from typing import Optional, List, Callable

import equinox as eqx
import equinox.nn as nn
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
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
    intermediate_size: int = eqx.static_field()
    act: Callable = eqx.static_field()
    c_fc: Gpt2Conv1D
    c_proj: Gpt2Conv1D
    dropout: nn.Dropout

    def __init__(self, config, intermediate_size, *, key):
        self.intermediate_size = intermediate_size
        embed_dim = config.hidden_size

        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = Gpt2Conv1D(out_features=intermediate_size, in_features=embed_dim, key=k_fc)
        self.c_proj = Gpt2Conv1D(out_features=embed_dim, in_features=intermediate_size, key=k_proj)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    @partial(jax.jit, static_argnames=["inference"])
    def __call__(self, hidden_states, *, inference: bool, key=None):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)
        return hidden_states


class Gpt2Attention(eqx.Module):
    config: GPT2Config = eqx.static_field()
    causal: bool = eqx.static_field()
    layer_idx: Optional[int]

    c_attn: Gpt2Conv1D
    c_proj: Gpt2Conv1D
    resid_dropout: nn.Dropout

    causal_mask: Optional[Array]

    @property
    def embed_dim(self):
        return self.config.hidden_size

    @property
    def num_heads(self):
        return self.config.num_attention_heads

    @property
    def head_dim(self):
        return self.embed_dim // self.num_heads

    def __init__(self, config: GPT2Config, *, key, layer_idx: Optional[int] = None, causal: bool = True):
        self.config = config
        assert self.embed_dim % self.num_heads == 0, f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
        self.causal = causal
        self.layer_idx = layer_idx

        k_c, k_q, k_proj = jrandom.split(key, 3)

        self.c_attn = Gpt2Conv1D(out_features=3 * self.embed_dim, in_features=self.embed_dim, key=k_c)
        self.c_proj = Gpt2Conv1D(out_features=self.embed_dim, in_features=self.embed_dim, key=k_proj)

        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

        if self.causal:
            self.causal_mask = jnp.tril(jnp.ones((config.n_positions, config.n_positions)))
        else:
            self.causal_mask = None

    # TODO: reorder_and_upcast_attn
    # TODO: scale_attn_by_inverse_layer_idx
    @partial(jax.jit, static_argnums=2)
    def __call__(self, hidden_states, inference: bool = True, *, key):
        # hidden_states has shape [seq_len, embed_dim]
        rng_key = key

        qkv_out = self.c_attn(hidden_states)  # [seq_len, 3 * embed_dim]
        query, key, value = jnp.split(qkv_out, 3, axis=1)  # [seq_len, embed_dim]

        query = self._split_heads(query)  # [num_heads, seq_len, head_dim]
        key = self._split_heads(key)
        value = self._split_heads(value)

        query_length, key_length = query.shape[1], key.shape[1]

        if self.causal:
            attention_mask = self.causal_mask[:query_length, :key_length]
        else:
            attention_mask = None

        key_d1, key_d2 = jrandom.split(rng_key, 2) if rng_key is not None else (None, None)
        attn_output = Gpt2Attention._multihead_attn(q=query, k=key, v=value, attention_mask=attention_mask)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, key=key_d2, inference=inference)

        return attn_output

    @staticmethod
    def _split_state(hidden_states: Array[..., "embed_dim"], num_split) -> Array[..., "num_split", "head_dim"]:
        embed_dim = hidden_states.shape[-1]
        assert embed_dim % num_split == 0
        return jnp.reshape(hidden_states, (-1, num_split, embed_dim // num_split))

    @staticmethod
    def _merge_state(hidden_states: Array[..., "num_split", "split_size"]) -> Array[..., "embed_dim"]:
        return jnp.reshape(hidden_states, (-1, hidden_states.shape[-2] * hidden_states.shape[-1]))

    def _split_heads(self, hidden_states: Array["seq_len", "embed_dim"]) -> Array["num_heads", "seq_len", "head_dim"]:
        # assert hidden_states.shape[-1] % self.num_heads == 0
        return jnp.transpose(Gpt2Attention._split_state(hidden_states, self.num_heads), axes=(1, 0, 2))

    @staticmethod
    def _merge_heads(hidden_states: Array["num_heads", "seq_len", "head_dim"]) -> Array["seq_len", "embed_dim"]:
        return Gpt2Attention._merge_state(jnp.transpose(hidden_states, axes=(1, 0, 2)))

    @staticmethod
    def _multihead_attn(q, k, v, attention_mask):
        # q, k, v have shape [heads, sequence, features]
        w = jnp.matmul(q, k.transpose([0, 2, 1]))
        w = w * lax.rsqrt(float(v.shape[-1]))

        w = Gpt2Attention._mask_attn_weights(w, attention_mask)
        w = jnn.softmax(w)
        a = jnp.matmul(w, v)
        return a

    @staticmethod
    def _mask_attn_weights(w, mask):
        if mask is not None:
            mask = jnp.broadcast_to(mask, w.shape)
            mask = lax.select(mask > 0, jnp.full(mask.shape, 0.0), jnp.full(mask.shape, -1e4))
            w = w + mask
        return w


class Gpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: Gpt2Attention
    ln_2: nn.LayerNorm
    mlp: Gpt2Mlp

    def __init__(self, config, layer_idx, *, key):
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(config, layer_idx=layer_idx, key=k_attn)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(config, inner_dim, key=k_mlp)

    @partial(jax.jit, static_argnums=2)
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
    wte: nn.Embedding
    wpe: nn.Embedding
    dropout: nn.Dropout
    blocks: List[Gpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: GPT2Config, *, key):
        super().__init__()

        k_wte, k_wpe, k_blocks = jrandom.split(key, 3)
        embed_dim = config.n_embd

        self.wte = nn.Embedding(
            config.vocab_size, embed_dim,
            weight=jrandom.normal(key=k_wte,
                                  shape=(config.vocab_size, embed_dim)) * config.initializer_range,
            key=k_wte,
        )

        self.wpe = nn.Embedding(config.max_position_embeddings, embed_dim,
            weight=jrandom.normal(
                key=k_wpe,
                shape=(config.max_position_embeddings, embed_dim)
            ) * config.initializer_range / 2,
            key=k_wpe,
        )
        self.dropout = nn.Dropout(p=config.embd_pdrop)
        self.blocks = [
            Gpt2Block(config, layer_idx=i, key=k) for i, k in enumerate(jrandom.split(k_blocks, config.n_layer))
        ]
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)

    @partial(jax.jit, static_argnums=(2))
    def __call__(self, input_ids: Array["seq_len"], inference=True, *, key):
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(jnp.arange(input_ids.shape[0], dtype="i4"))

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
            self.lm_head = self.transformer.wte.weight
        else:
            self.lm_head = jrandom.normal(k_lm_head,
                                          (config.vocab_size, config.hidden_size)) * config.initializer_range

    @partial(jax.jit, static_argnums=(2))
    def __call__(self, input_ids: Array["seq_len"], inference=True, *, key):
        hidden_states = self.transformer(input_ids, inference=inference, key=key)
        lm_logits = hidden_states @ jnp.transpose(self.lm_head)

        return lm_logits