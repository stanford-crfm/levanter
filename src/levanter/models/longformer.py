from typing import Optional

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
    assert Window.size < SeqLen.size, "Window size must be less than sequence length"
    windowed_key = hax.sliding_window(key, SeqLen, Window, -1000.0)
    windowed_value = hax.sliding_window(value, SeqLen, Window, 0.0)

    # we structurally can't attend to positions outside the window, but we also don't want to attend to initial padding
    windowed_mask = _ignore_padding_attn_mask(SeqLen, Window)
    mask = hax.nn.attention.combine_masks_and(mask, windowed_mask)

    # compute attention scores: contract out HeadDim, leaving SeqLen x Window
    # amazing that this works (I think it does!)
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
