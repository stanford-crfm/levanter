from typing import Optional

import jax.lax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from haliax.types import PrecisionLike


def causal_sliding_window_attention(
    SeqLen: Axis,
    Window: Axis,
    Head: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[NamedArray] = None,  # should conform to (Head, SeqLen, Window)
    bias: Optional[NamedArray] = None,  # should conform to (Head, SeqLen, Window)
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    Computes sliding window attention a la Longformer.
    """
    assert Window.size <= SeqLen.size, "Window size must be less than sequence length"
    windowed_key = hax.sliding_window(key, SeqLen, Window, -1000.0)
    windowed_value = hax.sliding_window(value, SeqLen, Window, 0.0)

    # we structurally can't attend to positions outside the window, but we also don't want to attend to initial padding
    windowed_mask = _ignore_padding_attn_mask(SeqLen, Window)
    mask = hax.nn.attention.combine_masks_and(mask, windowed_mask)

    return hax.nn.attention.dot_product_attention(
        SeqLen, Window, Head, query, windowed_key, windowed_value, mask, bias, attention_dtype, precision
    )


def _ignore_padding_attn_mask(SeqLen, Window):
    # initial padding is the first max(0, window_size - 1 - s) positions (where s is position in seq)
    # sanity check working througt it:
    # at s=0, we should only be able to attend to w=window_size-1
    # at s=window_size, we should be able to attend to 0<=w<=window_size-1
    # as s=k (for k <= window_size) we should be able to attend to window_size-k-1<=w<=window_size-1
    # as s=k (for k > window_size) we should be able to attend to 0<=w<=window_size-1
    return (Window.size - 1 - hax.arange(SeqLen)).broadcast_axis(Window) <= hax.arange(Window)


def causal_sliding_window_attention2(
    SeqLen: Axis,
    Window: Axis,
    Head: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[NamedArray] = None,  # should conform to (Head, SeqLen, Window)
    bias: Optional[NamedArray] = None,  # should conform to (Head, SeqLen, Window)
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    Computes sliding window attention a la Longformer. This method uses blocking because Jax can't figure it
    out automatically.
    """
    # We use the window size as the block size
    # The basic idea is that we want to compute attention one block (of query) at a time, where a block is a window
    # of the sequence. Each q can attend to the prior window_size-1 positions.

    assert Window.size <= SeqLen.size, "Window size must be less than sequence length"

    # TODO: relax?
    assert SeqLen.size % Window.size == 0, "Sequence length must be divisible by window size"

    Block = Axis("Block", SeqLen.size // Window.size)
    QWindow = Axis("QWindow", Window.size)

    rolled_key = hax.roll(key, Window.size, SeqLen)

    def attend_block(block_idx):
        block_start = block_idx * Window.size
        # our query block spans [block_start, block_start + window_size)
        query_block = _query_block(query, block_start)
        # extract the relevant window from the key and value
        key_block = _kv_block(key, block_start)
        value_block = _kv_block(value, block_start)

        # TODO: figure out mask
        return hax.nn.attention.dot_product_attention(
            QWindow, Window, Head, query_block, key_block, value_block, mask, bias, attention_dtype, precision
        )

    def _query_block(query, block_start):
        return query.take(SeqLen, hax.arange(QWindow, start=block_start))

    def _kv_block(kv, block_start):
        # this one is more complex: each q can attend to the prior window_size-1 positions
        # we want a tensor that is (..., QWindow, Window, Head) where each row is a window of kvs
        # which itself is conceptually a slice of the tensor sliding_window(kv, SeqLen, Window, -1000.0)






