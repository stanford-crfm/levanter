import math
from typing import List, Optional

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax
import haliax.random as hrandom
from haliax.core import NamedArray
from haliax.types import Axis, AxisSpec, PrecisionLike


# With attention we usually distinguish between the mask and the bias, though the former is just a special case of the
# latter. In practice, the mask is a boolean array that is applied after the softmax, while the bias is a float array
# that is applied before the softmax.

# because we use named axis we can be fairly loose about the shape of masks and biases: want to have a different
# mask for each head? fine. want to broadcast across the key sequence length? fine. etc etc


def dot_product_attention_weights(
    Head: Axis,
    KSeqLen: AxisSpec,
    query: NamedArray,
    key: NamedArray,
    mask: Optional[NamedArray] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. Computes the logits for the attention weights. Note that the
    "SeqLen" axis in query must be distinct from the "SeqLen" axis in key.

    :param Head: Axis of head dimension
    :param KSeqLen: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
    :param query: NamedArray of shape (QSeqLen, HeadDim)
    :param key: NamedArray of shape (KSeqLen, HeadDim)
    :param mask: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be boolean, applied after softmax.
    :param bias: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be float, applied before softmax.
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QSeqLen, KSeqLen)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L40
    import haliax.nn as hnn

    orig_dtype = query.dtype
    query = query / jnp.sqrt(Head.size)

    if attention_dtype is not None:
        query = query.astype(attention_dtype)
        key = key.astype(attention_dtype)

    weights = haliax.dot(Head, query, key, precision=precision)

    if bias is not None:
        weights = weights + bias

    weights = hnn.softmax(weights, axis=KSeqLen)

    if mask is not None:
        weights = weights * mask

    return weights.astype(orig_dtype)


def dot_product_attention(
    QSeqLen: Axis,
    KSeqLen: Axis,
    HeadDim: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[NamedArray] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. This can be multi-headed or not.

    :param QSeqLen: Axis of sequence length
    :param KSeqLen: Axis of key sequence length
    :param HeadDim: Axis of head dimension
    :param query: NamedArray of shape (QSeqLen, HeadDim)
    :param key: NamedArray of shape (KSeqLen, HeadDim)
    :param value: NamedArray of shape (KSeqLen, HeadDim)
    :param mask: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be boolean, applied after softmax.
    :param bias: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be float, applied before softmax.
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QSeqLen, HeadDim)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L125

    # rename key/value length axis if it's the same as the query length axis
    if KSeqLen == QSeqLen:
        KSeqLen = QSeqLen.alias(KSeqLen.name + "_key")
        key = key.rename({KSeqLen: QSeqLen})
        value = value.rename({KSeqLen: QSeqLen})

    weights = dot_product_attention_weights(HeadDim, KSeqLen, query, key, mask, bias, attention_dtype, precision)

    return haliax.dot(KSeqLen, weights, value)


def mask_to_bias(mask: NamedArray, mask_value: float = -1e9) -> NamedArray:
    return mask * mask_value


def combine_masks_and(mask1: Optional[NamedArray], mask2: Optional[NamedArray]) -> Optional[NamedArray]:
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 & mask2


def combine_masks_or(mask1: Optional[NamedArray], mask2: Optional[NamedArray]) -> Optional[NamedArray]:
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 | mask2


def causal_mask(QSeqLen: Axis, KSeqLen: Axis) -> NamedArray:
    """
    Creates a causal mask for attention.

    :param QSeqLen: Axis of query sequence length
    :param KSeqLen: Axis of key sequence length
    :return: NamedArray of shape (QSeqLen, KSeqLen)
    """
    # copilot wrote this and i'm just blown away
    return haliax.arange(QSeqLen).broadcast_axis(KSeqLen) >= haliax.arange(KSeqLen).broadcast_axis(QSeqLen)


def prefix_lm_mask(QSeqLen: Axis, KSeqLen: Axis, prefix_len: int) -> NamedArray:
    """Mask for the PrefixLM objective: fully connected before prefix_len, then causal after."""
    assert prefix_len >= 0
    assert prefix_len <= KSeqLen.size

    causal = causal_mask(QSeqLen, KSeqLen)
    prefix = haliax.arange(KSeqLen) < prefix_len

    return prefix | causal


def dropout_mask(axes: AxisSpec, dropout_rate: float, *, key: PRNGKey) -> NamedArray:
    """
    Really just an alias for haliax.random.bernoulli. You can pass in e.g. Head, QSeqLen and KSeqLen
    """
    return hrandom.bernoulli(key, shape=axes, p=1 - dropout_rate)


def forgetful_causal_mask(KSeqLen: Axis, mask_prob: float, sample_prob: bool = True, *, key: PRNGKey) -> NamedArray:
    """
    Forgetful Context Masking a la https://arxiv.org/abs/2210.13432. Randomly drops out positions from the key sequence.
    Reportedly better than normal attention dropout. Almost certainly faster.

    You're always allowed to attend to the 0th position. (They say BOS token, but we don't always start with bos)

    :param KSeqLen: Axis of key sequence length
    :param mask_prob: Probability a position to mask
    :param sample_prob: If True, sample the prob between 0 and the provided prob (this is what the paper does)
    """
    zeroth_on = haliax.nn.one_hot(0, KSeqLen, dtype=jnp.bool_)  # always allow 0th position
    if mask_prob == 0:
        return jnp.ones((KSeqLen.size,), dtype=jnp.bool_)
    elif mask_prob == 1:
        return zeroth_on
    else:
        if sample_prob:
            key, subkey = jax.random.split(key)
            mask_prob = jax.random.uniform(subkey, shape=(), minval=0, maxval=mask_prob)
        base: NamedArray = hrandom.bernoulli(key, shape=(KSeqLen,), p=1 - mask_prob)
        return base | zeroth_on


def _get_alibi_slopes(heads: int) -> List[float]:
    # from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
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


def alibi_attention_bias(Heads: Axis, SeqLen: Axis) -> NamedArray:
    """
    Creates an attention bias for alibi attention.

    :param SeqLen: Axis of sequence length
    :param Heads: Axis of heads
    :return: NamedArray of shape (Heads, QSeqLen)
    """
    slopes = haliax.named(jnp.array(_get_alibi_slopes(Heads.size)), Heads)
    positions = haliax.arange(SeqLen).broadcast_axis(Heads)

    return slopes * positions
