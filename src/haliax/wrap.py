import functools
from typing import Optional

import jax
import jax.numpy as jnp

from haliax.core import NamedArray, _broadcast_order, broadcast_to
from haliax.util import ensure_tuple

from .types import AxisSpec


def wrap_elemwise_unary(f):
    """Wraps a unary elementwise function to take and return NamedArrays"""

    @functools.wraps(f)
    def wrapper(a, *args, **kwargs):
        if isinstance(a, NamedArray):
            return NamedArray(f(a.array, *args, **kwargs), a.axes)
        else:
            return f(a, *args, **kwargs)

    return wrapper


def wrap_reduction_call(fn, single_axis_only: bool = False, supports_where: bool = True):
    @functools.wraps(fn)
    def wrapper(a, axis: Optional[AxisSpec] = None, where: NamedArray = None, **kwargs):
        kwargs = dict(kwargs)
        if where is not None and not supports_where:
            raise ValueError(f"where is not supported by {fn.__name__}")

        if kwargs.get("out", None) is not None:
            raise ValueError("out is not supported yet for NamedArray")
        if kwargs.get("keepdims", False):
            raise ValueError("keepdims is not supported for NamedArray")

        def reduce_one_leaf(a):
            nonlocal axis, where
            if isinstance(a, NamedArray):
                if where is not None:
                    if not isinstance(where, NamedArray):
                        raise TypeError("where must be a NamedArray if a is a NamedArray")
                    where = broadcast_to(where, a.axes)
                    kwargs["where"] = where.array

                if axis is None:
                    result = fn(a.array, axis=None, **kwargs)
                    if jnp.isscalar(result):
                        return result
                    else:
                        return NamedArray(result, ())
                else:
                    axis = ensure_tuple(axis)
                    if single_axis_only and len(axis) > 1:
                        raise ValueError(f"{fn.__name__} only supports a single axis")
                    indices = a._lookup_indices(axis)
                    if indices is None or any(x is None for x in indices):
                        raise ValueError(f"axis {axis} is not in {a.axes}")
                    new_axes = [ax for ax in a.axes if ax not in axis]
                    if single_axis_only:
                        result = fn(a.array, axis=indices[0], **kwargs)
                    else:
                        result = fn(a.array, axis=indices, **kwargs)
                    if jnp.isscalar(result):
                        return result
                    return NamedArray(result, tuple(new_axes))
            else:
                if where is not None:
                    kwargs["where"] = where
                return fn(a, axis=axis, **kwargs)

        return jax.tree_util.tree_map(reduce_one_leaf, a, is_leaf=lambda x: isinstance(x, NamedArray))

    wrapper.__doc__ = (
        """
    This function augments the behavior of `{fn}` to support NamedArrays, so that axis is a NamedArray.
    At the moment, neither `where` nor `out` are supported.
    =====

    """
        + fn.__doc__
    )
    return wrapper


def wrap_axiswise_call(fn, single_axis_only: bool):
    @functools.wraps(fn)
    def wrapper(a, axis: Optional[AxisSpec] = None, **kwargs):
        if isinstance(a, NamedArray):
            if axis is None:
                return NamedArray(fn(a.array, axis=None, **kwargs), a.axes)
            else:
                indices = ensure_tuple(a._lookup_indices(axis))
                if any(x is None for x in indices):
                    raise ValueError(f"axis {axis} is not in {a.axes}")
                if len(indices) == 1:
                    return NamedArray(fn(a.array, axis=indices[0], **kwargs), a.axes)
                elif single_axis_only:
                    raise ValueError(f"{fn.__name__} only supports a single axis")
                else:
                    return NamedArray(fn(a.array, axis=indices, **kwargs), a.axes)

        else:
            return fn(a, axis=axis, **kwargs)

    wrapper.__doc__ = (
        """
    This function augments the behavior of `{fn}` to support NamedArrays, so that axis is an Axis of sequence of axes.
    At the moment, neither `where` nor `out` are supported.
    =====

    """
        + fn.__doc__
    )
    return wrapper


def wrap_elemwise_binary(op):
    @functools.wraps(op)
    def binop(a, b):
        if isinstance(a, NamedArray) and isinstance(b, NamedArray):
            axes = _broadcast_order(a, b)
            a = broadcast_to(a, axes)
            b = broadcast_to(b, axes)
            return NamedArray(op(a.array, b.array), axes)
        elif isinstance(a, NamedArray) and jnp.isscalar(b):
            return NamedArray(op(a.array, b), a.axes)
        elif isinstance(b, NamedArray) and jnp.isscalar(a):
            return NamedArray(op(a, b.array), b.axes)
        else:
            return op(a, b)

    return binop


def unwrap_namedarrays(*a):
    return tuple(x.array if isinstance(x, NamedArray) else x for x in a)


__all__ = [
    "wrap_elemwise_unary",
    "wrap_reduction_call",
    "wrap_axiswise_call",
    "wrap_elemwise_binary",
    "unwrap_namedarrays",
]
