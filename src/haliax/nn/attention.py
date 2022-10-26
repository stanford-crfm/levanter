import math
from typing import List, Optional

import jax.numpy as jnp
from jax.random import PRNGKey

import haliax
import haliax.random as hrandom
from haliax import Axis, AxisSpec, NamedArray


def dot_product_attention_weights(
    HeadDim: Axis,
    KeySeqLen: AxisSpec,
    query: NamedArray,
    key: NamedArray,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. Computes the logits for the attention weights.

    :param HeadDim: Axis of head dimension
    :param KeySeqLen: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
    :param query: NamedArray of shape (SeqLen, HeadDim)
    :param key: NamedArray of shape (KeySeqLen, HeadDim)
    :param bias: Optional[NamedArray] broadcast compatible with (HeadDim, SeqLen, KeySeqLen)
    :param attention_dtype: Optional dtype to use for attention
    :return: NamedArray of shape (SeqLen, KeySeqLen)
    """
    import haliax.nn as hnn

    orig_dtype = query.dtype
    query = query / jnp.sqrt(HeadDim.size)

    if attention_dtype is not None:
        query = query.astype(attention_dtype)
        key = key.astype(attention_dtype)

    weights = haliax.dot(HeadDim, query, key)

    if bias is not None:
        weights = weights + bias

    weights = hnn.softmax(weights, axis=KeySeqLen)

    return weights.astype(orig_dtype)


def dot_product_attention(
    SeqLen: Axis,
    KeySeqLen: Axis,
    HeadDim: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention

    :param SeqLen: Axis of sequence length
    :param KeySeqLen: Axis of key sequence length
    :param HeadDim: Axis of head dimension
    :param query: NamedArray of shape (SeqLen, HeadDim)
    :param key: NamedArray of shape (KeySeqLen, HeadDim)
    :param value: NamedArray of shape (KeySeqLen, HeadDim)
    :param bias: Optional[NamedArray] broadcast compatible with (HeadDim, SeqLen, KeySeqLen)
    :param attention_dtype: Optional dtype to use for attention
    :return: NamedArray of shape (SeqLen, HeadDim)
    """

    # rename key/value length axis if it's the same as the query length axis
    if KeySeqLen == SeqLen:
        KeySeqLen = SeqLen.alias(KeySeqLen.name + "_key")
        key = key.rename({KeySeqLen: SeqLen})
        value = value.rename({KeySeqLen: SeqLen})

    weights = dot_product_attention_weights(HeadDim, KeySeqLen, query, key, bias, attention_dtype)

    return haliax.dot(KeySeqLen, weights, value)


def dropout_mask(SeqLen: Axis, KeySeqLen: Axis, dropout_rate: float, *, key: PRNGKey) -> NamedArray:
    return hrandom.bernoulli(key, (SeqLen, KeySeqLen), 1 - dropout_rate)


def mask_to_bias(mask: NamedArray, mask_value: float = -1e9) -> NamedArray:
    return mask * mask_value


def _get_alibi_slopes(heads: int) -> List[float]:
    # cf https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
    )


def alibi_attention_bias(SeqLen: Axis, Heads: Axis) -> NamedArray:
    """
    Creates an attention bias for alibi attention.

    :param SeqLen: Axis of sequence length
    :param Heads: Axis of heads
    :return: NamedArray of shape (Heads, SeqLen)
    """
    slopes = haliax.named(jnp.array(_get_alibi_slopes(Heads.size)), Heads)
    positions = haliax.arange(SeqLen).broadcast_axis(Heads)

    return slopes * positions
