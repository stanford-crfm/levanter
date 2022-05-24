from functools import partial
from typing import Callable, Any, Optional, TypeVar

import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn

from equinox.custom_types import Array
from jax import lax

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def fold_left(fn: Callable[[Carry, X], Carry], init: Carry, xs: X) -> Carry:
    res = lax.scan(lambda carry, x: (fn(carry, x), None), init=init, xs=xs)
    return res[0]


def quick_gelu(x):
    return x * jnn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(jnn.gelu, approximate=False),
    "relu": jnn.relu,
    "silu": jnn.silu,
    "swish": jnn.swish,
    "gelu_new": partial(jnn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}

# copied from https://github.com/google/flax/blob/5a5aa87a2e7abb0729d80be5c668a1e5620493ef/flax/linen/attention.py
# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable[..., Any] = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype = jnp.float32):
    """Mask-making helper for attention weights.
    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.
    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      pairwise_fn: broadcasting elementwise comparison function
      extra_batch_dims: number of extra batch dims to add singleton
        axes for, none by default
      dtype: mask return dtype
    Returns:
      A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = pairwise_fn(jnp.expand_dims(query_input, axis=-1),
                       jnp.expand_dims(key_input, axis=-2))
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype = jnp.float32) -> Array:
    """Make a causal mask for self-attention.
    In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
    will be `[batch..., heads, len, len]` and this function will produce a
    causal mask of shape `[batch..., 1, len, len]`.
    Args:
      x: input array of shape `[batch..., len]`
      extra_batch_dims: number of batch dims to add singleton axes for,
        none by default
      dtype: mask return dtype
    Returns:
      A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(idxs, idxs, jnp.greater_equal,
                               extra_batch_dims=extra_batch_dims, dtype=dtype)


def combine_masks(*masks: Optional[Array],
                  dtype=jnp.float32) -> Array:
    """Combine attention masks.
    Args:
      *masks: set of attention mask arguments to combine, some can be None.
      dtype: dtype for the returned mask.
    Returns:
      Combined mask, reduced by logical and, returns None if no masks given.
    """
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None
    assert all(map(lambda x: x.ndim == masks_list[0].ndim, masks_list)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks_list))}')
    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)

def dot_product_attention_weights(query: Array,
                                  key: Array,
                                  bias: Optional[Array] = None,
                                  mask: Optional[Array] = None,
                                  broadcast_dropout: bool = True,
                                  dropout_key = None,
                                  dropout_rate: float = 0.,
                                  deterministic: bool = False,
                                  dtype = jnp.float32,
                                  precision = None):
    """Computes dot-product attention weights given query and key.
    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.
    Args:
      query: queries for calculating attention with shape of
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: float32)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
    Returns:
      Output of shape `[batch..., num_heads, q_length, kv_length]`.
    """
    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], (
        'q, k batch dims must match.')
    assert query.shape[-2] == key.shape[-2], (
        'q, k num_heads must match.')
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key,
                              precision=precision)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    attn_weights = jnn.softmax(attn_weights).astype(dtype)

    # apply attention dropout
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = jrandom.bernoulli(dropout_key, keep_prob, dropout_shape)
        else:
            keep = jrandom.bernoulli(dropout_key, keep_prob, attn_weights.shape)
        multiplier = (keep.astype(attn_weights.dtype) /
                      jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    return attn_weights
