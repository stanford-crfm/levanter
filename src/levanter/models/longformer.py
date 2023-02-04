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
    PaddedLen = Axis("PaddedLen", SeqLen.size + Window.size - 1)

    padded_key = hax.pad_left(key, SeqLen, PaddedLen, 0.0)
    padded_value = hax.pad_left(value, SeqLen, PaddedLen, 0.0)

    # TODO: relax?
    assert SeqLen.size % Window.size == 0, "Sequence length must be divisible by window size"
    Block = Axis("Block", SeqLen.size // Window.size)
    BlockSize = Axis("BlockSize", Window.size * 2 - 1)

    # the attention structure is that each query attends to the prior window_size positions (q - window_size, q]
    # We extract one query block of length Window.size at a time (the block size and window size could be different, but
    # this seems fine)
    # For each query block, we extract a key and value block of length BlockSize == Window.size * 2 - 1
    # The key block is [query_block_start - window_size + 1, query_block_start + window_size)

    # for our attention masks, we have to account for the padding in the key and value blocks.
    # because of the padding
    # each q can attend to k s.t. k \in [q, q + window_size)
    # equivalently, k - q \in [0, window_size)
    diff = hax.arange(BlockSize) - hax.arange(Window).broadcast_axis(BlockSize)
    attn_mask = (diff >= 0) & (diff < Window.size)

    # no attending to padding for 0th block
    attn_mask_0 = attn_mask & (hax.arange(BlockSize) >= Window.size - 1)

    def attend_block(block_idx):
        block_idx = block_idx.scalar()
        block_start = block_idx * Window.size
        # our query block spans [block_start, block_start + window_size)
        query_block = query.slice(SeqLen, Window, start=block_start)
        # extract the relevant window from the key and value
        key_block = _kv_block(padded_key, block_start)
        value_block = _kv_block(padded_value, block_start)

        mask = jax.lax.cond(block_idx == 0, lambda _: attn_mask_0, lambda _: attn_mask, None)

        return hax.nn.attention.dot_product_attention(
            Window, BlockSize, Head, query_block, key_block, value_block, mask, None, attention_dtype, precision
        )

    def _kv_block(kv, block_start):
        # each q can attend to the prior window_size-1 positions
        return kv.slice(PaddedLen, BlockSize, start=block_start)

    # we use scan here to encourage jax to do the blocking
    _, blocked_attn = hax.scan(lambda _, block_idx: (None, attend_block(block_idx)), Block)(None, hax.arange(Block))  # type: ignore

    # now we need to unblock the attention
    return blocked_attn.flatten_axes((Block, Window), SeqLen)
