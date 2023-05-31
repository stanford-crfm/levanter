import math
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

import haliax
import haliax.random as hrandom
from haliax.core import NamedArray
from haliax.types import Axis, AxisSelection, AxisSpec, PrecisionLike


# With attention we usually distinguish between the mask and the bias, though the former is just a special case of the
# latter. In practice, the mask is a boolean array that is applied using `where` to the logits, while the bias is a
# float array that is added to the logits. The mask is usually used to prevent attention to certain positions, while
# the bias is usually used to encourage or discourage attention to certain positions.
# The mask usually is head-independent, while the bias is frequently head-dependent

# because we use named axis we can be fairly loose about the shape of masks and biases: want to have a different
# mask for each head? fine. want to broadcast across the key sequence length? fine. etc etc


def dot_product_attention_weights(
    Head: Axis,
    KPos: AxisSelection,
    query: NamedArray,
    key: NamedArray,
    mask: Optional[NamedArray] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. Computes the logits for the attention weights. Note that the
    "Pos" axis in query must be distinct from the "Pos" axis in key.

    :param Head: Axis of head dimension
    :param KPos: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
    :param query: NamedArray of shape (QPos, KeySize)
    :param key: NamedArray of shape (KPos, KeySize)
    :param mask: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be boolean
    :param bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QPos, KPos)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L40
    import haliax.nn as hnn

    orig_dtype = query.dtype
    query = query / jnp.sqrt(query.axis_size(Head))

    if attention_dtype is not None:
        query = query.astype(attention_dtype)
        key = key.astype(attention_dtype)

    weights = haliax.dot(Head, query, key, precision=precision)

    if bias is not None:
        weights = weights + bias
    if mask is not None:
        weights = haliax.where(mask, weights, -1e9)

    weights = hnn.softmax(weights, axis=KPos)

    return weights.astype(orig_dtype)


def dot_product_attention(
    QPos: Axis,
    KPos: Axis,
    KeySize: Axis,
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

    :param QPos: Axis of sequence length
    :param KPos: Axis of key sequence length
    :param KeySize: Axis of head dimension
    :param query: NamedArray of shape (QPos, KeySize)
    :param key: NamedArray of shape (KPos, KeySize)
    :param value: NamedArray of shape (KPos, KeySize)
    :param mask: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be boolean
    :param bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QPos, KeySize)

    Mask and bias are given as separate arguments because they are often computed separately and have different shapes.
    For example, mask is frequently just a boolean array of shape (QPos, KPos), while bias is frequently a float
    array of shape (KeySize, QPos, KPos) or (KeySize, KPos)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L125

    # rename key/value length axis if it's the same as the query length axis
    if KPos == QPos:
        KPos = QPos.alias(KPos.name + "_key")
        key = key.rename({KPos: QPos})
        value = value.rename({KPos: QPos})

    weights = dot_product_attention_weights(KeySize, KPos, query, key, mask, bias, attention_dtype, precision)

    return haliax.dot(KPos, weights, value)


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


def causal_mask(QPos: Axis, KPos: Axis) -> NamedArray:
    """
    Creates a causal mask for attention.

    :param QPos: Axis of query sequence length
    :param KPos: Axis of key sequence length
    :return: NamedArray of shape (QPos, KPos)
    """
    # copilot wrote this and i'm just blown away
    return haliax.arange(QPos).broadcast_axis(KPos) >= haliax.arange(KPos).broadcast_axis(QPos)


def dropout_mask(axes: AxisSpec, dropout_rate: float, *, key: PRNGKey) -> NamedArray:
    """
    Really just an alias for haliax.random.bernoulli. You can pass in e.g. Head, QPos and KPos
    """
    return hrandom.bernoulli(key, shape=axes, p=1 - dropout_rate)


def forgetful_causal_mask(KPos: Axis, mask_prob: float, sample_prob: bool = True, *, key: PRNGKey) -> NamedArray:
    """
    Forgetful Context Masking a la https://arxiv.org/abs/2210.13432. Randomly drops out positions from the key sequence.
    Reportedly better than normal attention dropout. Almost certainly faster.

    You're always allowed to attend to the 0th position. (They say BOS token, but we don't always start with bos)

    :param KPos: Axis of key sequence length
    :param mask_prob: Probability a position to mask
    :param sample_prob: If True, sample the prob between 0 and the provided prob (this is what the paper does)
    """
    zeroth_on = haliax.nn.one_hot(0, KPos, dtype=jnp.bool_)  # always allow 0th position
    if mask_prob == 0:
        return jnp.ones((KPos.size,), dtype=jnp.bool_)
    elif mask_prob == 1:
        return zeroth_on
    else:
        if sample_prob:
            key, subkey = jax.random.split(key)
            mask_prob = jax.random.uniform(subkey, shape=(), minval=0, maxval=mask_prob)
        base: NamedArray = hrandom.bernoulli(key, shape=(KPos,), p=1 - mask_prob)
        return base | zeroth_on


def _get_alibi_slopes(heads: int, bias_max: float) -> List[float]:
    # Mosaic supports "bias_max"
    log_bias_max = math.log2(bias_max)
    # from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - log_bias_max)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
    )


def alibi_attention_bias(Heads: Axis, KPos: Axis, bias_max: float = 8, dtype=jnp.float32) -> NamedArray:
    """
    Creates an attention bias for alibi attention.

    :param KPos: Axis of (key) sequence length
    :param Heads: Axis of heads
    :return: NamedArray of shape (Heads, KPos)
    """
    slopes = haliax.named(np.array(_get_alibi_slopes(Heads.size, bias_max)), Heads)
    positions = haliax.arange(KPos).broadcast_axis(Heads)

    biases = slopes * positions
    return biases.astype(dtype)
