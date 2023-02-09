import math
from typing import List, Optional

import jax
import jax.lax
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey

import haliax
import haliax as hax
import haliax.random as hrandom
from haliax.core import NamedArray
from haliax.types import Axis, AxisSpec, PrecisionLike


# With attention we usually distinguish between the mask and the bias, though the former is just a special case of the
# latter. In practice, the mask is a boolean array that is applied using `where` to the logits, while the bias is a
# float array that is added to the logits. The mask is usually used to prevent attention to certain positions, while
# the bias is usually used to encourage or discourage attention to certain positions.
# The mask usually is head-independent, while the bias is frequently head-dependent

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
    :param mask: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be boolean
    :param bias: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be float
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
    if mask is not None:
        weights = haliax.where(mask, weights, -1e9)

    weights = hnn.softmax(weights, axis=KSeqLen)

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
    :param mask: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be boolean
    :param bias: Optional[NamedArray] broadcast compatible with (HeadDim, QSeqLen, KSeqLen). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QSeqLen, HeadDim)

    Mask and bias are given as separate arguments because they are often computed separately and have different shapes.
    For example, mask is frequently just a boolean array of shape (QSeqLen, KSeqLen), while bias is frequently a float
    array of shape (HeadDim, QSeqLen, KSeqLen) or (HeadDim, KSeqLen)
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


def alibi_bias(Heads: Axis, SeqLen: Axis, dtype=jnp.float32) -> NamedArray:
    """
    Creates an attention bias for alibi attention.

    :param SeqLen: Axis of sequence length
    :param Heads: Axis of heads
    :return: NamedArray of shape (Heads, QSeqLen)
    """
    slopes = haliax.named(np.array(_get_alibi_slopes(Heads.size)), Heads)
    positions = haliax.arange(SeqLen).broadcast_axis(Heads)

    biases = slopes * positions
    return biases.astype(dtype)


def causal_sliding_window_attention(
    SeqLen: Axis,
    Window: Axis,
    Head: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    Computes sliding window attention a la Longformer. This method uses blocking because Jax can't figure it
    out automatically.
    """
    # We use the window size as the block size
    # The basic idea is that we want to compute attention one block (of query) at a time, where a block is a window
    # of the sequence. Each q can attend to the prior window_size-1 positions plus itself
    assert Window.size <= SeqLen.size, "Window size must be at least 2x sequence length"
    assert SeqLen.size % Window.size == 0, "Sequence length must be divisible by window size"

    if Window.size == SeqLen.size:
        # we can just use regular attention
        # we have to special case this because jax won't like the attend_block_N function
        # which doesn't actually get executed but does get traced
        K = SeqLen.alias("K")
        return hax.nn.attention.dot_product_attention(
            SeqLen,
            K,
            Head,
            query,
            key.rename({SeqLen: K}),
            value.rename({SeqLen: K}),
            mask=hax.nn.attention.causal_mask(SeqLen, K),
            bias=bias,
            attention_dtype=attention_dtype,
            precision=precision,
        )

    # the attention structure is that each query attends to the prior window_size positions (q - window_size, q]
    # We extract one query block of length Window.size at a time (the block size and window size could be different, but
    # this seems fine)
    # For each query block, we extract a key and value block of length KWindow == Window.size * 2 - 1
    # The key block is [query_block_start - window_size + 1, query_block_start + window_size)

    # TODO: relax?
    Block = Axis("Block", SeqLen.size // Window.size)
    KWindow = Axis("KWindow", Window.size * 2 - 1)  # this is what we need to grab from the key/value

    # this makes code a bit easier to read below
    Q = Window
    K = KWindow

    # for our attention masks, each q can attend to the prior window_size-1 positions plus itself
    # that is, each q can attend to k s.t. k \in [q - window_size + 1, q]
    # however, note that K is offset by window_size - 1, so we need to shift the mask by that amount
    # this means that we want to mask out k s.t. k \in [q, q + window_size)
    # equivalently, k - q \in [0, window_size)
    diff = hax.arange(K) - hax.arange(Q).broadcast_axis(K)
    attn_mask = (diff >= 0) & (diff < Window.size)

    def attend_block_N(block_idx):
        block_idx = block_idx.scalar()
        query_block = query.slice(SeqLen, Q, start=block_idx * Q.size)
        # extract the relevant window from the key and value
        # this spans [query_block_start - window_size + 1, query_block_start + window_size)
        key_block = key.slice(SeqLen, K, start=(block_idx - 1) * Q.size + 1)
        value_block = value.slice(SeqLen, K, start=(block_idx - 1) * Q.size + 1)

        if bias is not None:
            bias_block = bias.slice(SeqLen, K, start=(block_idx - 1) * Q.size + 1)
        else:
            bias_block = None

        return hax.nn.attention.dot_product_attention(
            Q, K, Head, query_block, key_block, value_block, attn_mask, bias_block, attention_dtype, precision
        )

    # for the 0th block, we have to worry about the out-of-bounds. just use a causal mask and do normal causal attention
    # NB if you change it so that the block size and window size aren't the same, you'll need to change this
    K0 = Q.alias("K0")
    attn_mask_0 = hax.nn.attention.causal_mask(Q, K0)

    def attend_block_0(block_idx):
        query_block = query.slice(SeqLen, Q, start=0)
        key_block = key.slice(SeqLen, K0, start=0)
        value_block = value.slice(SeqLen, K0, start=0)
        if bias is not None:
            bias_block = bias.slice(SeqLen, K0, start=0)
        else:
            bias_block = None
        return hax.nn.attention.dot_product_attention(
            Q, K0, Head, query_block, key_block, value_block, attn_mask_0, bias_block, attention_dtype, precision
        )

    # extra arg/return for dummy scan accumulator
    def attend_block(_, block_idx):
        return None, jax.lax.cond(block_idx.scalar() == 0, attend_block_0, attend_block_N, block_idx)

    # we use scan here to encourage jax to do the blocking
    _, blocked_attn = hax.scan(attend_block, Block)(None, hax.arange(Block))  # type: ignore

    # now we need to unblock the attention
    return blocked_attn.flatten_axes((Block, Q), SeqLen)
