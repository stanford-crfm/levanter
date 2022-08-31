from functools import wraps

import jax.numpy as jnp

from .core import Axis, NamedArray


@wraps(jnp.trace)
def trace(array: NamedArray, axis1: Axis, axis2: Axis, offset=0, dtype=None, out=None) -> NamedArray:
    a1_index = array._lookup_indices(axis1)
    a2_index = array._lookup_indices(axis2)

    if a1_index is None:
        raise ValueError(f"Axis {axis1} not found in array. Available axes: {array.axes}")
    if a2_index is None:
        raise ValueError(f"Axis {axis2} not found in array. Available axes: {array.axes}")

    if a1_index == a2_index:
        raise ValueError(f"Cannot trace along the same axis. Got {axis1} and {axis2}")

    if out is not None:
        raise NotImplementedError("out argument not supported")

    inner = jnp.trace(array.array, offset=offset, axis1=a1_index, axis2=a2_index, dtype=dtype, out=out)
    # remove the two indices
    axes = tuple(a for i, a in enumerate(array.axes) if i not in (a1_index, a2_index))
    return NamedArray(inner, axes)
