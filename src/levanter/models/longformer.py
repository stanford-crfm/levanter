from typing import Optional

import jax.lax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray

# from haliax.nn.attention import causal_mask
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


def _sliding_window_causal_mask(Window, KPos):
    diff = hax.arange(KPos) - hax.arange(Window).broadcast_axis(KPos)
    attn_mask = (diff >= 0) & (diff < Window.size)
    return attn_mask.rearrange((Window, KPos))


def causal_sliding_window_attention2(
    SeqLen: Axis,
    Window: Axis,
    Head: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
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
    assert Window.size <= SeqLen.size, "Window size must be less than sequence length"
    # TODO: relax?
    assert SeqLen.size % Window.size == 0, "Sequence length must be divisible by window size"
    Block = Axis("Block", SeqLen.size // Window.size)
    BlockSize = Axis(
        "BlockSize", Window.size * 2
    )  # * 2 because we want to include all possible positions for our query block

    # first things first: we roll the key and value tensors so that the first window_size-1 positions are
    # the last window_size-1 positions of the sequence
    # TODO: if we're clever we could avoid this and just do some indexing with a special case for the first block
    # unclear if that's better
    # -1 because we want to include the current position in the window
    rolled_key = hax.roll(key, Window.size - 1, SeqLen)
    rolled_value = hax.roll(value, Window.size - 1, SeqLen)

    # each w in Window attends to the next w positions (because we rolled the key and value)
    attn_mask = _sliding_window_causal_mask(Window, BlockSize)

    # for the 0th block, we don't want to attend to the first window_size-1 positions
    attn_mask_0 = attn_mask & (hax.arange(BlockSize) >= Window.size - 1)
    # for the final block, the key/value blocks will extend past the end,
    # so we need to mask out the extra positions and be careful with slicing
    # attn_mask_final = causal_mask(Window, BlockSize)

    def attend_block(block_idx):
        block_idx = block_idx.scalar()
        block_start = block_idx * Window.size
        # our query block spans [block_start, block_start + window_size)
        query_block = _query_block(query, block_start)
        # extract the relevant window from the key and value
        key_block = _kv_block(block_start, rolled_key)
        value_block = _kv_block(block_start, rolled_value)

        mask = jax.lax.cond(block_idx == 0, lambda _: attn_mask_0, lambda _: attn_mask, None)

        return hax.nn.attention.dot_product_attention(
            Window, BlockSize, Head, query_block, key_block, value_block, mask, None, attention_dtype, precision
        )

    def _query_block(query, block_start):
        return query.slice(SeqLen, Window, start=block_start)

    def _kv_block(block_start, rolled_kv):
        # each q can attend to the prior window_size-1 positions
        # because we rolled the key and value, we can just take the first BlockSize positions starting from block_start
        # we have to be careful for the final block: slice will mess with the start index if it's too large
        return rolled_kv.slice(SeqLen, BlockSize, start=block_start)

    # we use scan here to encourage jax to do the blocking
    _, blocked_attn = hax.scan(lambda _, block_idx: (None, attend_block(block_idx)), Block)(None, hax.arange(Block))  # type: ignore

    # now we need to unblock the attention
    return blocked_attn.flatten_axes((Block, Window), SeqLen)
