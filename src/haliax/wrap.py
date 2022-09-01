import functools
from typing import Optional

import jax.numpy as jnp

from haliax.core import AxisSpec, NamedArray, _broadcast_order, broadcast_to
from haliax.util import ensure_tuple


def wrap_elemwise_unary(f):
    """Wraps a unary elementwise function to take and return NamedArrays"""

    @functools.wraps(f)
    def wrapper(a, *args, **kwargs):
        if isinstance(a, NamedArray):
            return NamedArray(f(a.array, *args, **kwargs), a.axes)
        else:
            return f(a, *args, **kwargs)

    return wrapper


def wrap_reduction_call(fn):
    @functools.wraps(fn)
    def wrapper(a, axis: Optional[AxisSpec] = None, keepdims=False, **kwargs):
        if kwargs.get("where", None) is not None:
            raise ValueError("where is not supported yet for NamedArray")
        if kwargs.get("out", None) is not None:
            raise ValueError("where is not supported yet for NamedArray")

        if isinstance(a, NamedArray):
            if axis is None:
                result = fn(a.array, axis=None, keepdims=keepdims, **kwargs)
                if jnp.isscalar(result):
                    return result
                else:
                    return NamedArray(result, ())
            else:
                axis = ensure_tuple(axis)
                indices = a._lookup_indices(axis)
                if indices is None or any(x is None for x in indices):
                    raise ValueError(f"axis {axis} is not in {a.axes}")
                new_axes = a.axes if keepdims else [ax for ax in a.axes if ax not in axis]
                result = fn(a.array, axis=indices, keepdims=keepdims, **kwargs)
                if jnp.isscalar(result):
                    return result
                return NamedArray(fn(a.array, axis=indices, keepdims=keepdims, **kwargs), tuple(new_axes))
        else:
            return fn(a, axis=axis, keepdims=keepdims, **kwargs)

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


__all__ = ["wrap_elemwise_unary", "wrap_reduction_call", "wrap_axiswise_call", "wrap_elemwise_binary"]
