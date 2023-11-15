from typing import Optional

import jax.lax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, NamedArray
from haliax.types import PrecisionLike


def causal_sliding_window_attention(
    Pos: Axis,
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
    assert Window.size <= Pos.size, "Window size must be at least 2x sequence length"
    assert Pos.size % Window.size == 0, "Sequence length must be divisible by window size"

    if Window.size == Pos.size:
        # we can just use regular attention
        # we have to special case this because jax won't like the attend_block_N function
        # which doesn't actually get executed but does get traced
        K = Pos.alias("K")
        return hax.nn.attention.dot_product_attention(
            K,
            Head,
            query,
            key.rename({Pos: K}),
            value.rename({Pos: K}),
            mask=hax.nn.attention.causal_mask(Pos, K),
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
    Block = Axis("Block", Pos.size // Window.size)
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
        query_block = query.slice(Pos, Q, start=block_idx * Q.size)
        # extract the relevant window from the key and value
        # this spans [query_block_start - window_size + 1, query_block_start + window_size)
        key_block = key.slice(Pos, K, start=(block_idx - 1) * Q.size + 1)
        value_block = value.slice(Pos, K, start=(block_idx - 1) * Q.size + 1)

        if bias is not None:
            bias_block = bias.slice(Pos, K, start=(block_idx - 1) * Q.size + 1)
        else:
            bias_block = None

        return hax.nn.attention.dot_product_attention(
            K, Head, query_block, key_block, value_block, attn_mask, bias_block, attention_dtype, precision
        )

    # for the 0th block, we have to worry about the out-of-bounds. just use a causal mask and do normal causal attention
    # NB if you change it so that the block size and window size aren't the same, you'll need to change this
    K0 = Q.alias("K0")
    attn_mask_0 = hax.nn.attention.causal_mask(Q, K0)

    def attend_block_0(block_idx):
        query_block = query.slice(Pos, Q, start=0)
        key_block = key.slice(Pos, K0, start=0)
        value_block = value.slice(Pos, K0, start=0)
        if bias is not None:
            bias_block = bias.slice(Pos, K0, start=0)
        else:
            bias_block = None
        return hax.nn.attention.dot_product_attention(
            K0, Head, query_block, key_block, value_block, attn_mask_0, bias_block, attention_dtype, precision
        )

    # extra arg/return for dummy scan accumulator
    def attend_block(_, block_idx):
        return None, jax.lax.cond(block_idx.scalar() == 0, attend_block_0, attend_block_N, block_idx)

    # we use scan here to encourage jax to do the blocking
    _, blocked_attn = hax.scan(attend_block, Block)(None, hax.arange(Block))  # type: ignore

    # now we need to unblock the attention
    # TODO: see if the rearrange and flatten_axes have perf implications
    return blocked_attn.flatten_axes((Block, Q), Pos).rearrange(value.axes)
