# JAX wrapper around triton fused attention kernel
# https://github.com/openai/triton/blob/main/python/triton/ops/flash_attention.py
# Another example of calling triton kernels in JAX:
# https://github.com/jax-ml/jax-triton/blob/main/examples/fused_attention.py#L127

from typing import Tuple

import equinox
import jax
import jax_triton as jt
from triton.ops.flash_attention import attention

import haliax as hax
from haliax import NamedArray
from haliax.jax_utils import named_call


@named_call
def triton_flash_attention(
    q: NamedArray,
    k: NamedArray,
    v: NamedArray,
    softmax_scale: float = 1.0,
    causal: bool = True,
    sequence_parallel: bool = False,
) -> hax.NamedArray:

    output_shape = jax.ShapeDtypeStruct(shape=q.shape.values(), dtype=q.dtype)
    qkv = (q, k, v)

    return _triton_flash_attention(
        qkv=qkv,
        softmax_scale=softmax_scale,
        causal=causal,
        sequence_parallel=sequence_parallel,
        output_shape=output_shape,
    )


@equinox.filter_custom_vjp
def _triton_flash_attention(
    qkv: Tuple[hax.NamedArray, hax.NamedArray, hax.NamedArray],
    output_shape: jax.ShapeDtypeStruct,
    softmax_scale: float = 1.0,
    causal: bool = True,
    sequence_parallel: bool = False,
):
    """_triton_flash_attention exists (instead of applying the forward and backward
    functions directly to triton_flash_attention) because the custom_vjp decorator
    only differentiates the first argument of the function its applied to.

    Args:
        qkv (Tuple[hax.NamedArray, hax.NamedArray, hax.NamedArray]): _description_
        output_shape (jax.ShapeDtypeStruct): _description_
        softmax_scale (float, optional): _description_. Defaults to 1.0.
        causal (bool, optional): _description_. Defaults to True.
        sequence_parallel (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    q, k, v = qkv

    return _triton_flash_attention_forward(
        q=q,
        k=k,
        v=v,
        softmax_scale=softmax_scale,
        causal=causal,
        sequence_parallel=sequence_parallel,
        output_shape=output_shape,
    )


@named_call
def _triton_flash_attention_forward(
    q: NamedArray,
    k: NamedArray,
    v: NamedArray,
    output_shape: jax.ShapeDtypeStruct,
    softmax_scale: float = 1.0,
    causal: bool = True,
    sequence_parallel: bool = False,
):
    """Calls the Triton kernel implemention of flash attention.

    Args:
        q (NamedArray): _description_
        k (NamedArray): _description_
        v (NamedArray): _description_
        output_shape (jax.ShapeDtypeStruct): _description_
        softmax_scale (float, optional): _description_. Defaults to 1.0.
        causal (bool, optional): _description_. Defaults to True.
        sequence_parallel (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    print("\n\n MAKING CALL TO KERNEL NOW!\n\n")
    attn_output = jt.triton_call(
        q=q.array,
        k=k.array,
        v=v.array,
        causal=causal,
        sequence_parallel=sequence_parallel,
        sm_scale=softmax_scale,
        kernel=attention.forward,
        out_shape=output_shape,
        grid=None,
    )
    return hax.named(attn_output, q.axes)


@named_call
def _triton_flash_attention_backward(
    d_out: NamedArray,
):
    outshape = jax.ShapeDtypeStruct(shape=d_out.shape.values(), dtype=d_out.dtype)

    print("\n\nCALLING TRITON FLASH ATTEN BACKWARD\n\n")
    backward_out = jt.triton_call(
        do=d_out.array,
        kernel=attention.backward,
        out_shape=outshape,
        grid=None,
    )
    return hax.named(backward_out, d_out.axes)


_triton_flash_attention.def_fwd(_triton_flash_attention_forward)
_triton_flash_attention.def_bwd(_triton_flash_attention_backward)
