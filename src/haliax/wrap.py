import functools
from typing import Optional

import jax.numpy as jnp

from haliax.core import NamedArray, _ensure_tuple, AxisSpec


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
        if kwargs.get('where', None) is not None:
            raise ValueError("where is not supported yet for NamedArray")
        if kwargs.get('out', None) is not None:
            raise ValueError("where is not supported yet for NamedArray")

        if isinstance(a, NamedArray):
            if axis is None:
                result = fn(a.array, axis=None, keepdims=keepdims, **kwargs)
                if jnp.isscalar(result):
                    return result
                else:
                    return NamedArray(result, ())
            else:
                axis = _ensure_tuple(axis)
                indices = a.lookup_indices(axis)
                if indices is None or any(x is None for x in indices):
                    raise ValueError(f"axis {axis} is not in {a.axes}")
                new_axes = a.axes if keepdims else [ax for ax in a.axes if ax not in axis]
                result = fn(a.array, axis=indices, keepdims=keepdims, **kwargs)
                if jnp.isscalar(result):
                    return result
                return NamedArray(fn(a.array, axis=indices, keepdims=keepdims, **kwargs), new_axes)
        else:
            return fn(a, axis=axis, keepdims=keepdims, **kwargs)

    wrapper.__doc__ = """
    This function augments the behavior of `{fn}` to support NamedArrays, so that axis is a NamedArray.
    At the moment, neither `where` nor `out` are supported.
    =====

    """ + fn.__doc__
    return wrapper


def wrap_normalization_call(fn, single_axis_only: bool):
    @functools.wraps(fn)
    def wrapper(a, axis: Optional[AxisSpec] = None, **kwargs):
        if isinstance(a, NamedArray):
            if axis is None:
                return NamedArray(fn(a.array, axis=None, **kwargs), ())
            else:
                indices = _ensure_tuple(a.lookup_indices(axis))
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

    wrapper.__doc__ = """
    This function augments the behavior of `{fn}` to support NamedArrays, so that axis is a NamedArray.
    At the moment, neither `where` nor `out` are supported.
    =====

    """ + fn.__doc__
    return wrapper


__all__ = ["wrap_elemwise_unary", "wrap_reduction_call", "wrap_normalization_call"]
