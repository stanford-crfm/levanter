import functools
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

from psithuros import jax_utils
from psithuros.modeling_utils import ACT2FN


class Gpt2Conv1D(eqx.Module):
    kernel: Array
    bias: Array

    def __init__(self, *, in_features: int, out_features: int, key):
        self.kernel = jrandom.normal(key, [in_features, out_features]) * 0.02
        self.bias = jnp.zeros(out_features)

    def __call__(self, inputs):
        kernel = self.kernel
        return inputs @ kernel + self.bias


class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: Gpt2Conv1D
    c_proj: Gpt2Conv1D

    def __init__(self, config, intermediate_size, *, key):
        embed_dim = config.hidden_size

        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = Gpt2Conv1D(out_features=intermediate_size, in_features=embed_dim, key=k_fc)
        self.c_proj = Gpt2Conv1D(out_features=embed_dim, in_features=intermediate_size, key=k_proj)
        self.act = ACT2FN[config.activation_function]

    def __call__(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class Gpt2Attention(eqx.Module):
    causal: bool = eqx.static_field()
    head_dim: int = eqx.static_field()
    num_heads: int = eqx.static_field()

    c_attn: Gpt2Conv1D
    c_proj: Gpt2Conv1D
    dropout: pnn.Dropout

    @property
    def total_head_dim(self):
        return self.head_dim * self.num_heads

    def __init__(self, in_dim: int, num_heads: int, head_dim: int, dropout_prob: float, *, key, causal: bool = True):
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = head_dim

        k_c, k_proj = jrandom.split(key, 2)

        self.c_attn = Gpt2Conv1D(out_features=3 * self.total_head_dim, in_features=in_dim, key=k_c)
        self.c_proj = Gpt2Conv1D(out_features=in_dim, in_features=self.total_head_dim, key=k_proj)
        self.dropout = pnn.Dropout(dropout_prob)


    # TODO: cross-attention
    # TODO: reorder_and_upcast_attn
    # TODO: scale_attn_by_inverse_layer_idx
    # @eqx.filter_jit
    def __call__(self, hidden_states, inference: bool = True, *, key):
        # hidden_states has shape [seq_len, embed_dim]
        seq_len = hidden_states.shape[-2]
        rng_key = key

        qkv_out = self.c_attn(hidden_states)  # [seq_len, 3 * embed_dim]
        query, key, value = jnp.split(qkv_out, 3, axis=-1)  # [seq_len, embed_dim]

        query = self._split_heads(query)  # [num_heads, seq_len, head_dim]
        key = self._split_heads(key)
        value = self._split_heads(value)

        # must use negative indexing to please the pmap gods
        query_length, key_length = query.shape[-2], key.shape[-2]

        if self.causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            causal_mask = jnp.where(causal_mask, 0.0, -1E9)
            causal_mask = causal_mask.astype(jnp.bfloat16)
            attention_mask = causal_mask[:query_length, :key_length]
        else:
            attention_mask = None

        attn_weights = jnp.einsum('... n d, ... m d -> ... n m', query, key)  # [heads, seq_len, seq_len]
        attn_weights = attn_weights * lax.rsqrt(float(value.shape[-1]))

        if attention_mask is not None:
            # mask = jnp.broadcast_to(attention_mask, w.shape)
            # w = jnp.where(mask > 0, w, -1E9)
            attn_weights = attn_weights + attention_mask

        attn_weights = jnn.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights, key=rng_key, inference=inference)

        attn_output = jnp.einsum('... n m, ... m d -> ... n d', attn_weights, value)  # [heads, seq_len, head_dim]

        attn_output = self._merge_heads(attn_output)  # [seq_len, embed_dim]
        attn_output = self.c_proj(attn_output)

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
    resid_dropout: pnn.Dropout

    def __init__(self, config: GPT2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        head_dim = hidden_size // config.n_head

        assert hidden_size % config.n_head == 0, \
            f"embed_dim={hidden_size} must be divisible by num_heads={config.n_head}"

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(hidden_size, num_heads=config.n_head, head_dim=head_dim,
                                  dropout_prob=config.attn_pdrop, key=k_attn, causal=True)
        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(config, inner_dim, key=k_mlp)

    # @eqx.filter_jit
    def __call__(self, hidden_states, inference=True, *, key):
        k1, k2, k3 = jax_utils.maybe_rng_split(key, 3)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, inference=inference, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.mlp(hidden_states)
        ff_output = self.resid_dropout(ff_output, inference=inference, key=k3)

        hidden_states = ff_output + residual

        return hidden_states


class Gpt2Transformer(eqx.Module):
    config: GPT2Config = eqx.static_field()
    blocks: List[Gpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: GPT2Config, *, key):
        super().__init__()
        self.config = config

        embed_dim = config.n_embd

        self.blocks = [
            Gpt2Block(config, key=k) for i, k in enumerate(jrandom.split(key, config.n_layer))
        ]
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)

    # @eqx.filter_jit
    def __call__(self, hidden_states: Array["seq_len", "embed_dim"], inference=True, *, key):
        keys = jax_utils.maybe_rng_split(key, len(self.blocks))

        if inference:
            for block, k_block, i in zip(self.blocks, keys, range(len(self.blocks))):
                hidden_states = block(hidden_states, inference=inference, key=k_block)
        else:
            hidden_states = recursive_checkpoint([lambda x: block(x, inference=inference, key=k_block) for block, k_block in zip(self.blocks, keys)], threshold=2)(hidden_states)
            # for block, k_block, i in zip(self.blocks, keys, range(len(self.blocks))):
            #     print(i)
            #     hidden_states = jax.remat(Gpt2Model.block_remat)(block, hidden_states, key=k_block)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


# from https://github.com/google/jax/issues/4285
def recursive_checkpoint(funs, threshold = 2):
    if len(funs) == 1:
        return funs[0]
    elif len(funs) == 2:
        f1, f2 = funs
        return lambda x: f1(f2(x))
    elif len(funs) <= threshold:
        return functools.reduce(lambda f, g: lambda x: f(g(x)), funs)
    else:
        f1 = recursive_checkpoint(funs[:len(funs)//2])
        f2 = recursive_checkpoint(funs[len(funs)//2:])
        return lambda x: f1(jax.remat(f2)(x))


class Gpt2Embeddings(eqx.Module):
    token_embeddings: jnp.ndarray
    position_embeddings: jnp.ndarray
    token_out_embeddings: Optional[jnp.ndarray]
    dropout: pnn.Dropout

    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 num_position_embeddings: int,
                 initializer_range: float,
                 tie_word_embeddings: bool,
                 dropout_prob: float, *, key):
        super().__init__()
        k_wte, k_wpe, k_out = jrandom.split(key, 3)

        self.token_embeddings = jrandom.normal(key=k_wte,
                                               shape=(vocab_size, embed_dim)) * initializer_range
        self.position_embeddings = jrandom.normal(key=k_wpe,
                                                  shape=(num_position_embeddings,
                                                         embed_dim)) * initializer_range / 2
        self.dropout = pnn.Dropout(p=dropout_prob)

        if tie_word_embeddings:
            self.token_out_embeddings = None
        else:
            self.token_out_embeddings = jrandom.normal(key=k_out,
                                                       shape=(vocab_size, embed_dim)) * initializer_range

    def embed(self, input_ids: Array["seq_len"], inference, *, key):
        input_embeds = self.token_embeddings[input_ids]
        position_embeds = self.position_embeddings[jnp.arange(input_ids.shape[-1], dtype="i4")]
        hidden_states = input_embeds + position_embeds
        hidden_states = self.dropout(hidden_states, inference=inference, key=key)

        return hidden_states

    def unembed(self, hidden_states: Array["seq_len"]):
        embeddings = self.token_out_embeddings or self.token_embeddings
        return jnp.einsum('... l h, ... v h -> ... l v', hidden_states, embeddings)


class Gpt2LMHeadModel(eqx.Module):
    transformer: Gpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    def __init__(self, config: GPT2Config, *, key):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = Gpt2Transformer(config, key=k_t)
        self.embeddings = Gpt2Embeddings(vocab_size=config.vocab_size,
                                         embed_dim=config.n_embd,
                                         num_position_embeddings=config.n_positions,
                                         initializer_range=config.initializer_range,
                                         tie_word_embeddings=config.tie_word_embeddings,
                                         dropout_prob=config.embd_pdrop,
                                         key=k_embeddings)

    def __call__(self, input_ids: Array["seq_len"], key):
        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=key is None, key=k_embed)
        hidden_states = self.transformer(hidden_states, inference=key is None, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits
