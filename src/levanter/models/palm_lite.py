# from https://github.com/lucidrains/PaLM-jax/blob/dc76a8580a337d790b1c77908fc3052b87dc565c/palm_jax/palm_lite.py
from math import floor, log2
from typing import List, Tuple

import numpy as onp
from einops import rearrange, repeat
from equinox import Module, static_field
from jax import jit, lax, nn
from jax import numpy as np
from jax import random
from jax.numpy import einsum


# rmsnorm


class RMSNorm(Module):
    gamma: np.ndarray
    scale: float = static_field()
    eps: float = static_field()

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((dim,))
        self.eps = eps
        self.scale = dim**0.5

    def __call__(self, x):
        sum_of_squares = np.sum(np.square(x), axis=-1, keepdims=True)
        inv_norm = lax.rsqrt(sum_of_squares + self.eps)
        return inv_norm * x * self.gamma * self.scale


# AliBi


def get_alibi_slopes(heads):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if log2(heads).is_integer():
        return get_slopes_power_of_2(heads)
    closest_power_of_2 = 2 ** floor(log2(heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
    )


def calc_alibi_bias(seq_len, heads):
    slopes = get_alibi_slopes(heads)
    slopes = rearrange(onp.array(slopes), "h -> h 1 1")
    bias = rearrange(onp.arange(seq_len), "j -> 1 1 j")
    return slopes * bias


# attention - multi-query, one-headed key / values variant
# feedforward - Shazeer's SwiGLU variant


class ParallelTransformerBlock(Module):
    norm: Module
    wi: np.ndarray
    attn_wo: np.ndarray
    ff_wo: np.ndarray

    heads: int = static_field()
    fused_dims: Tuple[int] = static_field()
    scale: float = static_field()
    mask_value: float = static_field()

    def __init__(self, dim, dim_head, heads, key, ff_mult=4, mask_value=-1e10):
        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.norm = RMSNorm(dim)
        self.fused_dims = (
            attn_inner_dim,
            dim_head,
            dim_head,
            ff_inner_dim,
            ff_inner_dim,
        )

        self.wi = random.normal(key, (dim, sum(self.fused_dims)))
        self.attn_wo = random.normal(key, (attn_inner_dim, dim))
        self.ff_wo = random.normal(key, (ff_inner_dim, dim))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.mask_value = mask_value

    def __call__(self, x, *, attn_bias):
        _, split_indices = x.shape[-2], onp.cumsum(self.fused_dims[:-1])

        x = self.norm(x)

        # fused attention and feedforward projections
        q, k, v, ff, ff_gate = np.split(x @ self.wi, split_indices, axis=-1)
        # split out heads
        q = rearrange(q, "... n (h d) -> ... h n d", h=self.heads)

        # scale
        q *= self.scale

        # sim
        sim = einsum("... h i d, ... j d -> ... h i j", q, k)

        # causal mask
        sim = sim + attn_bias

        # attention
        attn = nn.softmax(sim, axis=-1)

        # aggregate values
        out = einsum("... h i j, ... j d -> ... h i d", attn, v)

        # merge heads
        out = rearrange(out, "... h n d -> ... n (h d)")

        # feedforward out
        attn_out = out @ self.attn_wo
        ff_out = (ff * nn.swish(ff_gate)) @ self.ff_wo

        # combine heads out
        return attn_out + ff_out


# main class


class PaLM(Module):
    embedding: np.ndarray
    norm: Module
    layers: List[Module]
    attn_bias: onp.ndarray

    def __init__(self, *, num_tokens, dim, dim_head, depth, heads, key, ff_mult=4, max_seq_len=2048, mask_value=-1e10):
        self.embedding = random.normal(key, (num_tokens, dim)) * 0.02

        causal_mask = onp.tril(onp.ones((max_seq_len, max_seq_len)))
        alibi_bias = calc_alibi_bias(max_seq_len, heads=heads)
        self.attn_bias = np.where(causal_mask, repeat(alibi_bias, "h 1 j -> h i j", i=max_seq_len), mask_value)

        self.layers = [
            ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, key=key, ff_mult=ff_mult)
            for _ in range(depth)
        ]
        self.norm = RMSNorm(dim)

    @jit
    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        attn_bias = self.attn_bias[..., :n, :n]

        for block in self.layers:
            x = block(x, attn_bias=attn_bias) + x

        x = self.norm(x)
        return x @ self.embedding.transpose()
