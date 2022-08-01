from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List

import equinox as eqx
import equinox.nn as nn
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec

import psithuros.nn as pnn
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from einops import rearrange

from psithuros import jax_utils
from psithuros.axis_names import Array

## very minimal gpt2 implementation just to get a sense of pjit
from psithuros.models.gpt2 import Gpt2Conv1D, Gpt2Config, Gpt2Attention, Gpt2Mlp, Gpt2Embeddings

resource_map = defaultdict(default_factory=lambda: None,
                           embed="model",
                           mlp="model")

resource_map.setdefault(None)


def layer_norm_init(shape):
    weight, bias = pjit(lambda shape: (jnp.ones(shape), jnp.zeros(shape)),
         in_axis_resources=None,
         out_axis_resources=resource_map["embed"])(shape)

    ln = nn.LayerNorm(shape)
    ln.weight = weight
    ln.bias = bias

    return ln


def conv1d_init(out_axis_resources, in_axis_resources, out_features, in_features, *, key):
    kernel, bias = pjit(lambda shape: (jrandom.normal(key, shape) * 0.02, jnp.zeros(shape[1])),
         in_axis_resources=None,
         out_axis_resources=(
             PartitionSpec(resource_map[in_axis_resources], resource_map[out_axis_resources]),
             PartitionSpec(resource_map[out_axis_resources])
         ))((in_features, out_features))

    conv = Gpt2Conv1D(in_features=in_features,
                      out_features=out_features,
                      key=key)

    assert conv.kernel.shape == kernel.shape
    assert conv.bias.shape == bias.shape
    conv.kernel = kernel
    conv.bias = bias

    return conv

def pjit_normal(shape, resource_shape, scale, *, key,):
    return pjit(lambda shape: jrandom.normal(key, shape) * scale,
                in_axis_resources=None,
                out_axis_resources=PartitionSpec(*[resource_map[s] for s in resource_shape]))(shape)


def embeddings_init(embed_dim: int,
                 vocab_size: int,
                 num_position_embeddings: int,
                 initializer_range: float,
                 tie_word_embeddings: bool,
                 dropout_prob: float, *, key):

    embeddings = Gpt2Embeddings(embed_dim=embed_dim,
                                vocab_size=vocab_size,
                                num_position_embeddings=num_position_embeddings,
                                initializer_range=initializer_range,
                                tie_word_embeddings=tie_word_embeddings,
                                dropout_prob=dropout_prob,
                                key=key,)
    k_wte, k_wpe, k_out = jrandom.split(key, 3)

    token_embeddings = pjit_normal(shape=(vocab_size, embed_dim),
                                   resource_shape=("vocab", "embed"),
                                   scale=initializer_range,
                                   key=k_wte)

    position_embeddings = pjit_normal(shape=(num_position_embeddings, embed_dim),
                                      resource_shape=("position", "embed"),
                                      scale=initializer_range,
                                      key=k_wpe)

    assert embeddings.token_embeddings.shape == token_embeddings.shape
    assert embeddings.position_embeddings.shape == position_embeddings.shape
    embeddings.token_embeddings = token_embeddings
    embeddings.position_embeddings = position_embeddings

    if not tie_word_embeddings:
        token_out_embeddings = pjit_normal(shape=(vocab_size, embed_dim),
                                            resource_shape=("vocab", "embed"),
                                            scale=initializer_range,
                                            key=k_out)
        assert embeddings.token_out_embeddings.shape == token_out_embeddings.shape
        embeddings.token_out_embeddings = token_out_embeddings

    return embeddings

def gpt2_mlp_init(embed_dim, intermediate_size, activation_fn, *, key):
    mlp = Gpt2Mlp(embed_dim, intermediate_size, activation_fn, key=key)  # type: ignore
    k_fc, k_proj = jrandom.split(key, 2)
    mlp.c_fc = conv1d_init(
        out_axis_resources=resource_map["mlp"],
        in_axis_resources=resource_map["embed"],
        out_features=intermediate_size,
        in_features=embed_dim,
        key=k_fc),
    mlp.c_proj = conv1d_init(
        out_axis_resources=resource_map["embed"],
        in_axis_resources=resource_map["mlp"],
        out_features=embed_dim,
        in_features=intermediate_size,
        key=k_proj)

    return mlp


def gpt2_attention_init(in_dim, num_heads, head_dim, dropout_prob, *, key)->Gpt2Attention:
    attn = Gpt2Attention(in_dim, num_heads=num_heads, head_dim=head_dim, dropout_prob=dropout_prob, key=k_attn, causal=True)  # type: ignore

    k_attn, k_proj = jrandom.split(key, 3)
    attn.c_attn = conv1d_init(
        out_axis_resources=resource_map["mlp"],
        in_axis_resources=resource_map["embed"],
        out_features=3 * attn.total_head_dim,
        in_features=in_dim, key=k_attn)
    attn.c_proj = conv1d_init(
        out_axis_resources=resource_map["embed"],
        in_axis_resources=resource_map["mlp"],
        in_features=attn.total_head_dim,
        out_features=in_dim, key=k_proj)

    return attn


class PjitGpt2Block(eqx.Module):
    ln_1: nn.LayerNorm
    attn: Gpt2Attention
    ln_2: nn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: pnn.Dropout

    def __init__(self,
                 config: Gpt2Config,
                 key: Optional[jax.random.PRNGKey] = None):
        super().__init__()
        k_attn, k_cross, k_mlp = jrandom.split(key, 3)

        hidden_size = config.hidden_dim
        inner_dim = 4 * hidden_size
        head_dim = hidden_size // config.num_heads

        assert hidden_size % config.num_heads == 0, \
            f"embed_dim={hidden_size} must be divisible by num_heads={config.num_heads}"

        self.ln_1 = layer_norm_init((hidden_size,))
        self.attn = gpt2_attention_init(hidden_size, config.num_heads, head_dim, config.dropout_prob, key=k_attn)
        self.ln_2 = layer_norm_init((hidden_size,))
        self.mlp = gpt2_mlp_init(embed_dim=hidden_size, intermediate_size=inner_dim, activation_fn=config.activation_function, key=k_mlp)

        self.resid_dropout = pnn.Dropout(config.resid_pdrop)

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


class PjitGpt2Transformer(eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: List[PjitGpt2Block]
    ln_f: nn.LayerNorm

    def __init__(self, config: Gpt2Config, *, key: jax.random.PRNGKey):
        super().__init__()
        self.config = config
        self.blocks = [
            PjitGpt2Block(config, key=k) for i, k in enumerate(jrandom.split(key, config.num_layers))
        ]
        self.ln_f = layer_norm_init((config.hidden_dim,))

    def __call__(self, hidden_states, inference=True, *, key):

        #TODO: add back in checkpoints
        for block in self.blocks:
            hidden_states = block(hidden_states, inference=inference, key=key)

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class PjitGpt2LMHeadModel(eqx.Module):
    transformer: PjitGpt2Transformer
    embeddings: Gpt2Embeddings

    @property
    def config(self):
        return self.transformer.config

    def __init__(self, vocab_size: int, config: Gpt2Config, *, key):
        k_t, k_embeddings = jrandom.split(key, 2)
        self.transformer = PjitGpt2Transformer(config, key=k_t)
        self.embeddings = embeddings_init(vocab_size=vocab_size,
                                          embed_dim=config.hidden_dim,
                                          num_position_embeddings=config.seq_len,
                                          initializer_range=config.initializer_range,
                                          tie_word_embeddings=True,
                                          dropout_prob=config.embed_pdrop,
                                          key=k_embeddings)

    def __call__(self, input_ids: Array["seq_len"], key):
        k_embed, k_transformer = jax_utils.maybe_rng_split(key, 2)
        hidden_states = self.embeddings.embed(input_ids, inference=key is None, key=k_embed)
        hidden_states = self.transformer(hidden_states, inference=key is None, key=k_transformer)
        lm_logits = self.embeddings.unembed(hidden_states)

        return lm_logits



