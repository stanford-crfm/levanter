from typing import Tuple

import equinox
import jax
import jax.numpy as jnp
import jax_triton as jt
from triton import cdiv
from triton.ops.flash_attention import _bwd_kernel, _bwd_preprocess, _fwd_kernel

import haliax as hax


def triton_flash_attention(
    q: hax.NamedArray,
    k: hax.NamedArray,
    v: hax.xNamedArray,
    softmax_scale: float = 1.0,
    causal: bool = True,
):
    out_shape = jax.ShapeDtypeStruct(shape=q.shape.values(), dtype=q.dtype)
    qkv = (q, k, v)

    return _triton_flash_attention(
        qkv,
        out_shape=out_shape,
        softmax_scale=softmax_scale,
        causal=causal,
    )


@equinox.filter_custom_vjp
def _triton_flash_attention(
    qkv: Tuple[hax.NamedArray, hax.NamedArray, hax.NamedArray],
    out_shape: jax.ShapeDtypeStruct,
    softmax_scale: float = 1.0,
    causal: bool = True,
    sequence_parallel: bool = False,
):
    return _triton_flash_attention_forward(
        None,
        qkv,
        out_shape=out_shape,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _triton_flash_attention_forward(
    ignore,
    qkv,
    out_shape: jax.ShapeDtypeStruct,
    softmax_scale: float = 1.0,
    causal: bool = True,
):
    del ignore
    q, k, v = qkv

    # only support for Ampere now
    BLOCK_M = 128
    BLOCK_N = 64

    # shape constraints
    Lq, Lk, Lv = q.shape["head_size"], k.shape["head_size"], v.shape["head_size"]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    # Init output placeholers
    out = jnp.empty_like(q)
    L = jnp.empty((q.shape["batch"] * q.shape["heads"], q.shape["position"]), device=q.device, dtype=jnp.float32)

    grid = (cdiv(q.shape["position"], BLOCK_M), q.shape["batch"] * q.shape["heads"], 1)
    num_warps = 4 if Lk <= 64 else 8

    attn_output = jt.triton_call(
        q.array,
        k.array,
        v.array,
        softmax_scale,
        L,
        out,
        q.array.stride(0),
        q.array.stride(1),
        q.array.stride(2),
        q.array.stride(3),
        k.array.stride(0),
        k.array.stride(1),
        k.array.stride(2),
        k.array.stride(3),
        v.array.stride(0),
        v.array.stride(1),
        v.array.stride(2),
        v.array.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        q.shape["batch"],
        q.shape["heads"],
        q.shape["position"],
        q.shape["batch"] * q.shape["heads"] * q.shape["position"],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=Lk,
        IS_CAUSAL=causal,
        num_warps=num_warps,
        num_stages=4,
        kernel=_fwd_kernel,
        out_shape=out_shape,
        grid=grid,
    )

    return hax.named(attn_output, q.axes), (out, L, Lk)


def _triton_flash_attention_backward(
    residuals,
    grad_in: hax.NamedArray,
    ignore,
    qkv,
    out_shape: jax.ShapeDtypeStruct,
    softmax_scale: float = 1.0,
    causal: bool = True,
):
    del ignore
    out, L, Lk = residuals
    q, k, v = qkv
    d_out = grad_in

    BLOCK = 128
    MMA_V3 = False
    seq_len_kv = k.shape["position"]
    d_out = d_out.contiguous()
    out_shape = jax.ShapeDtypeStruct(shape=q.shape.values(), dtype=q.dtype)

    # Create output objects
    dq = jnp.zeros_like(q, dtype=q.dtype)
    dk = jnp.empty_like(k)
    dv = jnp.empty_like(v)
    delta = jnp.empty_like(L)
    bwd_preprocess_grid = (cdiv(q.shape["position"], BLOCK) * q.shape["batch"] * q.shape["heads"],)

    jt.triton_call(
        out,
        d_out,
        delta,
        BLOCK_M=BLOCK,
        D_HEAD=Lk,
        kernel=_bwd_preprocess,
        out_shape=out_shape,
        grid=bwd_preprocess_grid,
    )

    bwd_grid = (q.shape["batch"] * q.shape["heads"], 1)

    jt.triton_call(
        q.array,
        k.array,
        v.array,
        softmax_scale,
        out,
        d_out,
        dq,
        dk,
        dv,
        L,
        delta,
        out.array.numel(),
        q.array.stride(0),
        q.array.stride(1),
        q.array.stride(2),
        q.array.stride(3),
        k.array.stride(0),
        k.array.stride(1),
        k.array.stride(2),
        k.array.stride(3),
        v.array.stride(0),
        v.array.stride(1),
        v.array.stride(2),
        v.array.stride(3),
        q.shape["batch"],
        q.shape["head"],
        q.shape["position"],
        q.shape["batch"] * q.shape["head"] * q.shape["position"],
        cdiv(seq_len_kv, BLOCK) * q.shape["batch"] * q.shape["head"] * q.shape["position"],
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        SEQUENCE_PARALLEL=False,
        CAUSAL=causal,
        MMA_V3=MMA_V3,
        num_warps=8,
        num_stages=1,
        kernel=_bwd_kernel,
        grid=bwd_grid,
    )

    if len(dq.shape) == 5:
        dq = dq.sum(dim=0)

    return dq, dk, dv, None, None, None


_triton_flash_attention.def_fwd(_triton_flash_attention_forward)
_triton_flash_attention.def_bwd(_triton_flash_attention_backward)
