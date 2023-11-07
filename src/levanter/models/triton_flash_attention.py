# JAX wrapper around triton fused attention kernel
# https://github.com/openai/triton/blob/main/python/triton/ops/flash_attention.py
# Another example of calling triton kernels in JAX:
# https://github.com/jax-ml/jax-triton/blob/main/examples/fused_attention.py#L127

import functools

import jax
import jax_triton as jt
from triton.ops.flash_attention import attention

from haliax import NamedArray


@functools.partial(jax.jit, static_argnames=["softmax_scale"])
def triton_flash_attention(
    q: NamedArray,
    k: NamedArray,
    v: NamedArray,
    softmax_scale: float = 1.0,
    causal: bool = True,
    sequence_parallel: bool = False,
):
    """_summary_

    Args:
        q (NamedArray): _description_
        k (NamedArray): _description_
        v (NamedArray): _description_
        softmax_scale (float, optional): _description_. Defaults to 1.0.
        causal (bool, optional): _description_. Defaults to True.
        sequence_parallel (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    print(q)
    print(q.shape)
    print(tuple(q.shape.values()))

    outshape = tuple(q.shape.values())

    return jt.triton_call(
        q=q,
        k=k,
        v=v,
        causal=causal,
        sequence_parallel=sequence_parallel,
        sm_scale=softmax_scale,
        kernel=attention,
        out_shape=outshape,
    )
