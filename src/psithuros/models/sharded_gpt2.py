from typing import List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange
from transformers import GPT2Config

import psithuros.nn as pnn
from psithuros import jax_utils
from psithuros.models.gpt2 import Gpt2Mlp, Gpt2Attention, Gpt2Embeddings
from psithuros.named_tensors import *

SHARD = "shard"

# We use model sharding in two different ways:
# For embeddings, we just split the embedding into multiple shards and use all_gather to concatenate them.
# For the transformer blocks:
#  * For attention: each shard handles num_heads/num_shards heads, each producing an embed_size embedding.
#    We then sum the embeddings via psum
# * For MLP: each mlp shard projects from embed_size to inner_size/num_shards, then to embed_size. We then sum the
#   projections via psum.


class ShardedGpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: Shaped[SHARD, Gpt2Attention]
    resid_dropout: pnn.Dropout
    ln_2: nn.LayerNorm
    mlp: Shaped[SHARD, Gpt2Mlp]

    def __init__(self, config: GPT2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        num_shards = jax.lax.psum(1, axis_name=SHARD)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        head_dim = hidden_size // config.n_head
        num_heads_per_shard = config.n_head // num_shards

        assert hidden_size % config.n_head == 0, \
            f"embed_dim={hidden_size} must be divisible by num_heads={config.n_head}"

        assert config.n_head % num_shards == 0, \
            f"num_heads={config.n_head} must be divisible by num_shards={num_shards}"

        assert inner_dim % num_shards == 0, \
            f"inner_dim={inner_dim} must be divisible by num_shards={num_shards}"

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(hidden_size, num_heads=num_heads_per_shard, head_dim=head_dim,
                                  dropout_prob=config.attn_pdrop, key=k_attn, causal=True)  # type: ignore
        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(config, inner_dim // num_shards, key=k_mlp)

    def __call__(self, hidden_states, inference=True, *, key):
        k1, k2, k3 = jax_utils.maybe_rng_split(key, 3)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, inference=inference, key=k1)
        # sum out the shard dimension. each shard has the whole hidden state now
        attn_output = jax.lax.psum(attn_output, axis_name=SHARD)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.mlp(hidden_states)
        ff_output = jax.lax.psum(ff_output, axis_name=SHARD)
        # sum out the shard dimension. each shard has the whole hidden state now
        ff_output = self.resid_dropout(ff_output, inference=inference, key=k3)

        hidden_states = ff_output + residual

        return hidden_states


class ShardedGpt2Transformer(eqx.Module):
    config: GPT2Config = eqx.static_field()
    blocks: List[ShardedGpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: GPT2Config, *, key):
        super().__init__()
        self.config = config

        embed_dim = config.n_embd

        self.blocks = [
            ShardedGpt2Block(config, key=k) for i, k in enumerate(jrandom.split(key, config.n_layer))
        ]
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)

    # @eqx.filter_jit
    def __call__(self, hidden_states: Array["seq_len", "embed_dim"], inference=True, *, key):
        keys = jax_utils.maybe_rng_split(key, len(self.blocks))

        for block, k_block, i in zip(self.blocks, keys, range(len(self.blocks))):
            hidden_states = block(hidden_states, inference=inference, key=k_block)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class ShardedGpt2LMHeadModel(eqx.Module):
    transformer: ShardedGpt2Transformer
    embeddings: Shaped[SHARD, Gpt2Embeddings]
    num_shards: int = eqx.static_field()
    embed_size_per_shard: int = eqx.static_field()

    @property
    def config(self):
        return self.transformer.config

    def __init__(self, config: GPT2Config, *, key: Shaped[SHARD, jrandom.PRNGKey]):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = ShardedGpt2Transformer(config, key=k_t)
        num_shards = jax.lax.psum(1, axis_name=SHARD)
        self.num_shards = num_shards

        assert config.n_embd % num_shards == 0, \
            f"embed_dim={config.n_embd} must be divisible by num_shards={num_shards}"

        self.embed_size_per_shard = config.n_embd // num_shards

        self.embeddings = Gpt2Embeddings(vocab_size=config.vocab_size,
                                         embed_dim=self.embed_size_per_shard,
                                         num_position_embeddings=config.n_positions,
                                         initializer_range=config.initializer_range,
                                         tie_word_embeddings=config.tie_word_embeddings,
                                         dropout_prob=config.embd_pdrop,
                                         key=k_embeddings)

    def __call__(self, input_ids: Array["seq_len"], key):
        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        # my_shard = jax.lax.axis_index(SHARD)
        my_shard = 0

        hidden_states = self.embeddings.embed(input_ids, inference=key is None, key=k_embed)
        # doesn't work because of https://github.com/google/jax/issues/11193
        # hidden_states = jax.lax.all_gather(hidden_states, axis_name=SHARD, tiled=True, axis=-1)

        # this doesn't work either
        # hidden_states = jax.lax.all_gather(hidden_states, axis_name=SHARD, axis=-1)
        # hidden_states = jnp.reshape(hidden_states, hidden_states.shape[:-2] + (-1,))

        # nor this
        # hidden_states = jax.lax.all_to_all(jax.lax.broadcast(hidden_states, (self.num_shards,)), SHARD, 0, -1, tiled=True)
        # hidden_states = jnp.reshape(hidden_states, hidden_states.shape[1:])


        print(hidden_states.shape)

        local_hidden_states = hidden_states

        full_size = hidden_states.shape[0:-1] + (self.config.n_embd, )
        hidden_states = jnp.zeros(full_size, dtype=hidden_states.dtype)
        hidden_states = jax.lax.dynamic_update_slice_in_dim(hidden_states, local_hidden_states, jnp.array(my_shard * self.embed_size_per_shard), -1)
        # hidden_states.at[my_shard:my_shard+self.embed_size_per_shard].set(local_hidden_states, indices_are_sorted=True)
        hidden_states = jax.lax.psum(hidden_states, axis_name=SHARD)

        # hidden_states = jax.lax.scatter(hidden_states, my_shard, hidden_states[my_shard], axis_name=SHARD)


        hidden_states = self.transformer(hidden_states, inference=key is None, key=k_transformer)

        # I don't love this, but we have to re-shard the hidden states by slicing followed up by a psum
        hidden_states = rearrange(hidden_states, "... (s h) -> ... s h", s=self.num_shards)[..., my_shard, :]
        lm_logits = self.embeddings.unembed(hidden_states)
        lm_logits = jax.lax.psum(lm_logits, axis_name=SHARD)

        return lm_logits






