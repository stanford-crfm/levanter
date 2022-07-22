from typing import List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange

import psithuros.nn as pnn
from psithuros import jax_utils
from psithuros.models.gpt2 import Gpt2Mlp, Gpt2Attention, Gpt2Embeddings, recursive_checkpoint, Gpt2Config
from psithuros.axis_names import *

# We use model sharding in two different ways:
# For embeddings, we just split the embedding into multiple shards and use all_gather to concatenate them.
# For the transformer blocks:
#  * For attention: each shard handles num_heads/num_shards heads, each producing an embed_size embedding.
#    We then sum the embeddings via psum
# * For MLP: each mlp shard projects from embed_size to inner_size/num_shards, then to embed_size. We then sum the
#   projections via psum.


class ShardedGpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: Shaped[LogicalAxis.PARAMS, Gpt2Attention]
    resid_dropout: pnn.Dropout
    ln_2: nn.LayerNorm
    mlp: Shaped[LogicalAxis.PARAMS, Gpt2Mlp]

    def __init__(self, config: Gpt2Config, *, key):
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        num_shards = jax.lax.psum(1, axis_name=LogicalAxis.PARAMS)
        hidden_size = config.hidden_dim
        inner_dim = 4 * hidden_size
        head_dim = hidden_size // config.num_heads
        num_heads_per_shard = config.num_heads // num_shards

        assert hidden_size % config.num_heads == 0, \
            f"embed_dim={hidden_size} must be divisible by num_heads={config.num_heads}"

        assert config.num_heads % num_shards == 0, \
            f"num_heads={config.num_heads} must be divisible by num_shards={num_shards}"

        assert inner_dim % num_shards == 0, \
            f"inner_dim={inner_dim} must be divisible by num_shards={num_shards}"

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(hidden_size, num_heads=num_heads_per_shard, head_dim=head_dim,
                                  dropout_prob=config.attn_pdrop, key=k_attn, causal=True)  # type: ignore
        self.resid_dropout = pnn.Dropout(p=config.resid_pdrop)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = Gpt2Mlp(hidden_size, inner_dim // num_shards, config.activation_function, key=k_mlp)

    def __call__(self, hidden_states, inference=True, *, key):
        k1, k2, k3 = jax_utils.maybe_rng_split(key, 3)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, inference=inference, key=k1)
        # sum out the shard dimension. each shard has the whole hidden state now
        attn_output = jax.lax.psum(attn_output, axis_name=LogicalAxis.PARAMS)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.mlp(hidden_states)
        ff_output = jax.lax.psum(ff_output, axis_name=LogicalAxis.PARAMS)
        # sum out the shard dimension. each shard has the whole hidden state now
        ff_output = self.resid_dropout(ff_output, inference=inference, key=k3)

        hidden_states = ff_output + residual

        return hidden_states


class ShardedGpt2Transformer(eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: List[ShardedGpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: Gpt2Config, *, key):
        super().__init__()
        self.config = config

        embed_dim = config.hidden_dim

        self.blocks = [
            ShardedGpt2Block(config, key=k) for i, k in enumerate(jrandom.split(key, config.num_layers))
        ]
        self.ln_f = nn.LayerNorm(embed_dim, eps=config.layer_norm_epsilon)

    # @eqx.filter_jit
    def __call__(self, hidden_states: Array["seq_len", "embed_dim"], inference=True, *, key):
        keys = jax_utils.maybe_rng_split(key, len(self.blocks))

        if True:
            for block, k_block, i in zip(self.blocks, keys, range(len(self.blocks))):
                hidden_states = block(hidden_states, inference=inference, key=k_block)
        else:
            hidden_states = recursive_checkpoint(
                [lambda x: block(x, inference=inference, key=k_block) for block, k_block in zip(self.blocks, keys)],
                threshold=2)(hidden_states)

        hidden_states = self.ln_f(hidden_states)

        return hidden_states


class ShardedGpt2LMHeadModel(eqx.Module):
    transformer: ShardedGpt2Transformer
    embeddings: Shaped[LogicalAxis.PARAMS, Gpt2Embeddings]
    num_shards: int = eqx.static_field()
    embed_size_per_shard: int = eqx.static_field()

    @property
    def config(self):
        return self.transformer.config

    def __init__(self, vocab_size: int, config: Gpt2Config, *, key: Shaped[LogicalAxis.PARAMS, jrandom.PRNGKey]):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = ShardedGpt2Transformer(config, key=k_t)
        num_shards = jax.lax.psum(1, axis_name=LogicalAxis.PARAMS)
        self.num_shards = num_shards

        assert config.hidden_dim % num_shards == 0, \
            f"embed_dim={config.hidden_dim} must be divisible by num_shards={num_shards}"

        self.embed_size_per_shard = config.hidden_dim // num_shards

        self.embeddings = Gpt2Embeddings(vocab_size=vocab_size,
                                         embed_dim=self.embed_size_per_shard,
                                         num_position_embeddings=config.seq_len,
                                         initializer_range=config.initializer_range,
                                         tie_word_embeddings=True,
                                         dropout_prob=config.embed_pdrop,
                                         key=k_embeddings)

    def __call__(self, input_ids: Array["seq_len"], key):
        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        my_shard = jax.lax.axis_index(LogicalAxis.PARAMS)

        hidden_states = self.embeddings.embed(input_ids, inference=key is None, key=k_embed)  # each shard has [seq_len, embed_dim//num_shards]
        hidden_states = self._concatenate_shards(hidden_states, my_shard)  # each shard has [seq_len, embed_dim]

        hidden_states = self.transformer(hidden_states, inference=key is None, key=k_transformer)  # still [seq_len, embed_dim]

        # I don't love this, but we have to re-shard the hidden states by slicing followed up by a psum
        hidden_states = rearrange(hidden_states, "... (s h) -> ... s h", s=self.num_shards)[..., my_shard, :]  # now [seq_len, embed_dim//num_shards]
        # each embedding shard is a [vocab_size x embed_dim//num_shards] matrix
        lm_logits = self.embeddings.unembed(hidden_states)  # now [seq_len, vocab_size], but only with information from the current shard
        lm_logits = jax.lax.psum(lm_logits, axis_name=LogicalAxis.PARAMS) # now [seq_len, vocab_size], with all shards

        return lm_logits

    def _concatenate_shards(self, hidden_states, my_shard_index):
        # doesn't work because of https://github.com/google/jax/issues/11193
        # hidden_states = jax.lax.all_gather(hidden_states, axis_name=LogicalAxis.PARAMS, tiled=True, axis=-1)

        # this doesn't work either
        # hidden_states = jax.lax.all_gather(hidden_states, axis_name=LogicalAxis.PARAMS, axis=-1)
        # hidden_states = jnp.reshape(hidden_states, hidden_states.shape[:-2] + (-1,))

        # nor this
        # hidden_states = jax.lax.all_to_all(jax.lax.broadcast(hidden_states, (self.num_shards,)), LogicalAxis.PARAMS, 0, -1, tiled=True)
        # hidden_states = jnp.reshape(hidden_states, hidden_states.shape[1:])

        local_hidden_states = hidden_states
        full_size = hidden_states.shape[0:-1] + (self.config.hidden_dim,)
        hidden_states = jnp.zeros(full_size, dtype=hidden_states.dtype)
        hidden_states = jax.lax.dynamic_update_slice_in_dim(hidden_states, local_hidden_states,
                                                            jnp.array(my_shard_index * self.embed_size_per_shard), -1)
        # hidden_states.at[my_shard:my_shard+self.embed_size_per_shard].set(local_hidden_states, indices_are_sorted=True)
        hidden_states = jax.lax.psum(hidden_states, axis_name=LogicalAxis.PARAMS)
        return hidden_states






