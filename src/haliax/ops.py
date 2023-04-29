from typing import Union

import jax
import jax.numpy as jnp

from .core import NamedArray, NamedOrNumeric, broadcast_arrays, broadcast_arrays_and_return_axes
from .types import Axis, AxisSelector


def trace(array: NamedArray, axis1: AxisSelector, axis2: AxisSelector, offset=0, dtype=None) -> NamedArray:
    """Compute the trace of an array along two named axes."""
    a1_index = array._lookup_indices(axis1)
    a2_index = array._lookup_indices(axis2)

    if a1_index is None:
        raise ValueError(f"Axis {axis1} not found in array. Available axes: {array.axes}")
    if a2_index is None:
        raise ValueError(f"Axis {axis2} not found in array. Available axes: {array.axes}")

    if a1_index == a2_index:
        raise ValueError(f"Cannot trace along the same axis. Got {axis1} and {axis2}")

    inner = jnp.trace(array.array, offset=offset, axis1=a1_index, axis2=a2_index, dtype=dtype)
    # remove the two indices
    axes = tuple(a for i, a in enumerate(array.axes) if i not in (a1_index, a2_index))
    return NamedArray(inner, axes)


def where(condition: Union[NamedOrNumeric, bool], x: NamedOrNumeric, y: NamedOrNumeric) -> NamedArray:
    """Like jnp.where, but with named axes. This version currently only accepts the three argument form."""

    # TODO: support the one argument form
    # if (x is None) != (y is None):
    #     raise ValueError("Must either specify both x and y, or neither")

    if jnp.isscalar(condition):
        if x is None:
            raise ValueError("Must specify x and y when condition is a scalar")
        return jax.lax.cond(condition, lambda _: x, lambda _: y, None)

    assert isinstance(condition, NamedArray)

    if jnp.isscalar(x):
        x = NamedArray(jnp.broadcast_to(x, condition.array.shape), condition.axes)

    assert isinstance(x, NamedArray)

    if jnp.isscalar(y):
        y = NamedArray(jnp.broadcast_to(y, condition.array.shape), condition.axes)

    assert isinstance(y, NamedArray)

    condition, x, y = broadcast_arrays(condition, x, y)  # type: ignore
    return NamedArray(jnp.where(condition.array, x.array, y.array), condition.axes)


def clip(array: NamedOrNumeric, a_min: NamedOrNumeric, a_max: NamedOrNumeric) -> NamedArray:
    """Like jnp.clip, but with named axes. This version currently only accepts the three argument form."""
    (array, a_min, a_max), axes = broadcast_arrays_and_return_axes(array, a_min, a_max)
    array = raw_array_or_scalar(array)
    a_min = raw_array_or_scalar(a_min)
    a_max = raw_array_or_scalar(a_max)

    return NamedArray(jnp.clip(array, a_min, a_max), axes)


def tril(array: NamedArray, axis1: Axis, axis2: Axis, k=0) -> NamedArray:
    """Compute the lower triangular part of an array along two named axes."""
    array = array.rearrange((..., axis1, axis2))

    inner = jnp.tril(array.array, k=k)
    return NamedArray(inner, array.axes)


def triu(array: NamedArray, axis1: Axis, axis2: Axis, k=0) -> NamedArray:
    """Compute the upper triangular part of an array along two named axes."""
    array = array.rearrange((..., axis1, axis2))

    inner = jnp.triu(array.array, k=k)
    return NamedArray(inner, array.axes)


def isclose(a: NamedArray, b: NamedArray, rtol=1e-05, atol=1e-08, equal_nan=False) -> NamedArray:
    """Returns a boolean array where two arrays are element-wise equal within a tolerance."""
    a, b = broadcast_arrays(a, b)
    # TODO: numpy supports an array atol and rtol, but we don't yet
    return NamedArray(jnp.isclose(a.array, b.array, rtol=rtol, atol=atol, equal_nan=equal_nan), a.axes)


def pad_left(array: NamedArray, axis: Axis, new_axis: Axis, value=0) -> NamedArray:
    """Pad an array along named axes."""
    amount_to_pad_to = new_axis.size - axis.size
    if amount_to_pad_to < 0:
        raise ValueError(f"Cannot pad {axis} to {new_axis}")

    idx = array._lookup_indices(axis)

    padding = [(0, 0)] * array.ndim
    if idx is None:
        raise ValueError(f"Axis {axis} not found in array. Available axes: {array.axes}")
    padding[idx] = (amount_to_pad_to, 0)

    padded = jnp.pad(array.array, padding, constant_values=value)
    return NamedArray(padded, array.axes[:idx] + (new_axis,) + array.axes[idx + 1 :])


def raw_array_or_scalar(x: NamedOrNumeric):
    if isinstance(x, NamedArray):
        return x.array
    return x


__all__ = ["trace", "where", "tril", "triu", "isclose", "pad_left", "clip"]
