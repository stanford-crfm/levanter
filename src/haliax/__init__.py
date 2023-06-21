from typing import Optional, Protocol, Sequence

import jax
import jax.numpy as jnp

import haliax.random as random
from haliax import nn as nn
from haliax import tree_util as tree_util

from .core import (
    NamedArray,
    are_shape_checks_enabled,
    broadcast_arrays,
    broadcast_axis,
    broadcast_to,
    concat_axis_specs,
    dot,
    enable_shape_checks,
    flatten_axes,
    named,
    rearrange,
    rename,
    roll,
    slice,
    slice_nd,
    split,
    take,
    unbind,
    unflatten_axis,
)
from .hof import fold, scan, vmap
from .ops import clip, isclose, pad_left, trace, tril, triu, where
from .partitioning import auto_sharded, axis_mapping, named_jit, shard_with_axis_mapping
from .types import Axis, AxisSelection, AxisSelector, AxisSpec
from .wrap import wrap_axiswise_call, wrap_elemwise_binary, wrap_elemwise_unary, wrap_reduction_call


# creation routines
def zeros(shape: AxisSpec, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 0"""
    if dtype is None:
        dtype = jnp.float32
    return full(shape, 0, dtype)


def ones(shape: AxisSpec, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 1"""
    if dtype is None:
        dtype = jnp.float32
    return full(shape, 1, dtype)


def full(shape: AxisSpec, fill_value, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to `fill_value`"""
    if isinstance(shape, Axis):
        return NamedArray(jnp.full(shape=shape.size, fill_value=fill_value, dtype=dtype), (shape,))
    else:
        x_shape = tuple(x.size for x in shape)
        return NamedArray(jnp.full(shape=x_shape, fill_value=fill_value, dtype=dtype), tuple(shape))


def zeros_like(a: NamedArray, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 0"""
    return NamedArray(jnp.zeros_like(a.array, dtype=dtype), a.axes)


def ones_like(a: NamedArray, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 1"""
    return NamedArray(jnp.ones_like(a.array, dtype=dtype), a.axes)


def full_like(a: NamedArray, fill_value, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to `fill_value`"""
    return NamedArray(jnp.full_like(a.array, fill_value, dtype=dtype), a.axes)


def arange(axis: Axis, *, start=0, step=1, dtype=None) -> NamedArray:
    """Version of jnp.arange that returns a NamedArray"""
    stop = start + axis.size * step
    return NamedArray(jnp.arange(start, stop, step, dtype=dtype), (axis,))


def stack(axis: AxisSelector, arrays: Sequence[NamedArray]) -> NamedArray:
    """Version of jnp.stack that returns a NamedArray"""
    if isinstance(axis, str):
        axis = Axis(axis, len(arrays))
    if len(arrays) == 0:
        return zeros(axis)
    arrays = [a.rearrange(arrays[0].axes) for a in arrays]
    return NamedArray(jnp.stack([a.array for a in arrays], axis=0), (axis,) + arrays[0].axes)


def concatenate(axis: AxisSelector, arrays: Sequence[NamedArray]) -> NamedArray:
    """Version of jnp.concatenate that returns a NamedArray"""
    total_size = _sum(a.resolve_axis(axis).size for a in arrays)
    if isinstance(axis, str):
        axis = Axis(axis, total_size)
    elif total_size != axis.size:
        raise ValueError(
            f"Cannot concatenate arrays along axis {axis.name} of size {axis.size} with total size {total_size}"
        )

    if len(arrays) == 0:
        return zeros(axis)

    arrays = [a.rearrange(arrays[0].axes) for a in arrays]
    axis_index = arrays[0]._lookup_indices(axis.name)

    if axis_index is None:
        raise ValueError(f"Axis {axis.name} not found in 0th array {arrays[0]}")

    new_axes = arrays[0].axes[:axis_index] + (axis,) + arrays[0].axes[axis_index + 1 :]
    return NamedArray(jnp.concatenate([a.array for a in arrays], axis=axis_index), new_axes)


# elementwise unary operations
abs = wrap_elemwise_unary(jnp.abs)
absolute = wrap_elemwise_unary(jnp.absolute)
angle = wrap_elemwise_unary(jnp.angle)
arccos = wrap_elemwise_unary(jnp.arccos)
arccosh = wrap_elemwise_unary(jnp.arccosh)
arcsin = wrap_elemwise_unary(jnp.arcsin)
arcsinh = wrap_elemwise_unary(jnp.arcsinh)
arctan = wrap_elemwise_unary(jnp.arctan)
arctanh = wrap_elemwise_unary(jnp.arctanh)
around = wrap_elemwise_unary(jnp.around)
bitwise_not = wrap_elemwise_unary(jnp.bitwise_not)
cbrt = wrap_elemwise_unary(jnp.cbrt)
ceil = wrap_elemwise_unary(jnp.ceil)
conj = wrap_elemwise_unary(jnp.conj)
conjugate = wrap_elemwise_unary(jnp.conjugate)
copy = wrap_elemwise_unary(jnp.copy)
cos = wrap_elemwise_unary(jnp.cos)
cosh = wrap_elemwise_unary(jnp.cosh)
deg2rad = wrap_elemwise_unary(jnp.deg2rad)
degrees = wrap_elemwise_unary(jnp.degrees)
exp = wrap_elemwise_unary(jnp.exp)
exp2 = wrap_elemwise_unary(jnp.exp2)
expm1 = wrap_elemwise_unary(jnp.expm1)
fabs = wrap_elemwise_unary(jnp.fabs)
fix = wrap_elemwise_unary(jnp.fix)
floor = wrap_elemwise_unary(jnp.floor)
frexp = wrap_elemwise_unary(jnp.frexp)
i0 = wrap_elemwise_unary(jnp.i0)
imag = wrap_elemwise_unary(jnp.imag)
invert = wrap_elemwise_unary(jnp.invert)
iscomplex = wrap_elemwise_unary(jnp.iscomplex)
isfinite = wrap_elemwise_unary(jnp.isfinite)
isinf = wrap_elemwise_unary(jnp.isinf)
isnan = wrap_elemwise_unary(jnp.isnan)
isneginf = wrap_elemwise_unary(jnp.isneginf)
isposinf = wrap_elemwise_unary(jnp.isposinf)
isreal = wrap_elemwise_unary(jnp.isreal)
log = wrap_elemwise_unary(jnp.log)
log10 = wrap_elemwise_unary(jnp.log10)
log1p = wrap_elemwise_unary(jnp.log1p)
log2 = wrap_elemwise_unary(jnp.log2)
logical_not = wrap_elemwise_unary(jnp.logical_not)
ndim = wrap_elemwise_unary(jnp.ndim)
negative = wrap_elemwise_unary(jnp.negative)
positive = wrap_elemwise_unary(jnp.positive)
rad2deg = wrap_elemwise_unary(jnp.rad2deg)
radians = wrap_elemwise_unary(jnp.radians)
real = wrap_elemwise_unary(jnp.real)
reciprocal = wrap_elemwise_unary(jnp.reciprocal)
rint = wrap_elemwise_unary(jnp.rint)
round = wrap_elemwise_unary(jnp.round)
rsqrt = wrap_elemwise_unary(jax.lax.rsqrt)  # nb this is in lax
sign = wrap_elemwise_unary(jnp.sign)
signbit = wrap_elemwise_unary(jnp.signbit)
sin = wrap_elemwise_unary(jnp.sin)
sinc = wrap_elemwise_unary(jnp.sinc)
sinh = wrap_elemwise_unary(jnp.sinh)
square = wrap_elemwise_unary(jnp.square)
sqrt = wrap_elemwise_unary(jnp.sqrt)
tan = wrap_elemwise_unary(jnp.tan)
tanh = wrap_elemwise_unary(jnp.tanh)
trunc = wrap_elemwise_unary(jnp.trunc)

# Reduction functions


class ReductionFunction(Protocol):
    def __call__(
        self, array: NamedArray, axis: Optional[AxisSelection] = None, where: Optional[NamedArray] = None, **kwargs
    ) -> NamedArray:
        ...


class SimpleReductionFunction(Protocol):
    def __call__(self, array: NamedArray, axis: Optional[AxisSelector] = None, **kwargs) -> NamedArray:
        ...


all: ReductionFunction = wrap_reduction_call(jnp.all)
amax: ReductionFunction = wrap_reduction_call(jnp.amax)
any: ReductionFunction = wrap_reduction_call(jnp.any)
argmax: SimpleReductionFunction = wrap_reduction_call(jnp.argmax, single_axis_only=True, supports_where=False)
argmin: SimpleReductionFunction = wrap_reduction_call(jnp.argmin, single_axis_only=True, supports_where=False)
max: ReductionFunction = wrap_reduction_call(jnp.max)
mean: ReductionFunction = wrap_reduction_call(jnp.mean)
min: ReductionFunction = wrap_reduction_call(jnp.min)
prod: ReductionFunction = wrap_reduction_call(jnp.prod)
ptp: ReductionFunction = wrap_reduction_call(jnp.ptp)
product: ReductionFunction = wrap_reduction_call(jnp.product)
sometrue: ReductionFunction = wrap_reduction_call(jnp.sometrue)
std: ReductionFunction = wrap_reduction_call(jnp.std)
_sum = sum
sum: ReductionFunction = wrap_reduction_call(jnp.sum)
var: ReductionFunction = wrap_reduction_call(jnp.var)


# "Normalization" functions that use an axis but don't change the shape
cumsum = wrap_axiswise_call(jnp.cumsum, True)
cumprod = wrap_axiswise_call(jnp.cumprod, True)
cumproduct = wrap_axiswise_call(jnp.cumproduct, True)
sort = wrap_axiswise_call(jnp.sort, True)
argsort = wrap_axiswise_call(jnp.argsort, True)

# elemwise binary ops
add = wrap_elemwise_binary(jnp.add)
arctan2 = wrap_elemwise_binary(jnp.arctan2)
bitwise_and = wrap_elemwise_binary(jnp.bitwise_and)
bitwise_or = wrap_elemwise_binary(jnp.bitwise_or)
bitwise_xor = wrap_elemwise_binary(jnp.bitwise_xor)
divide = wrap_elemwise_binary(jnp.divide)
divmod = wrap_elemwise_binary(jnp.divmod)
equal = wrap_elemwise_binary(jnp.equal)
float_power = wrap_elemwise_binary(jnp.float_power)
floor_divide = wrap_elemwise_binary(jnp.floor_divide)
fmax = wrap_elemwise_binary(jnp.fmax)
fmin = wrap_elemwise_binary(jnp.fmin)
fmod = wrap_elemwise_binary(jnp.fmod)
greater = wrap_elemwise_binary(jnp.greater)
greater_equal = wrap_elemwise_binary(jnp.greater_equal)
hypot = wrap_elemwise_binary(jnp.hypot)
left_shift = wrap_elemwise_binary(jnp.left_shift)
less = wrap_elemwise_binary(jnp.less)
less_equal = wrap_elemwise_binary(jnp.less_equal)
logaddexp = wrap_elemwise_binary(jnp.logaddexp)
logaddexp2 = wrap_elemwise_binary(jnp.logaddexp2)
logical_and = wrap_elemwise_binary(jnp.logical_and)
logical_or = wrap_elemwise_binary(jnp.logical_or)
logical_xor = wrap_elemwise_binary(jnp.logical_xor)
maximum = wrap_elemwise_binary(jnp.maximum)
minimum = wrap_elemwise_binary(jnp.minimum)
mod = wrap_elemwise_binary(jnp.mod)
multiply = wrap_elemwise_binary(jnp.multiply)
nextafter = wrap_elemwise_binary(jnp.nextafter)
not_equal = wrap_elemwise_binary(jnp.not_equal)
power = wrap_elemwise_binary(jnp.power)
remainder = wrap_elemwise_binary(jnp.remainder)
right_shift = wrap_elemwise_binary(jnp.right_shift)
subtract = wrap_elemwise_binary(jnp.subtract)
true_divide = wrap_elemwise_binary(jnp.true_divide)


__all__ = [
    "Axis",
    "NamedArray",
    "AxisSpec",
    "AxisSelection",
    "AxisSelector",
    "broadcast_to",
    "broadcast_axis",
    "named",
    "dot",
    "roll",
    "split",
    "flatten_axes",
    "slice",
    "slice_nd",
    "take",
    "unbind",
    "rename",
    "rearrange",
    "concat_axis_specs",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "full_like",
    "arange",
    "random",
    "abs",
    "absolute",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "around",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copy",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "fix",
    "floor",
    "frexp",
    "i0",
    "imag",
    "iscomplex",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "ndim",
    "negative",
    "positive",
    "rad2deg",
    "radians",
    "real",
    "reciprocal",
    "rint",
    "rsqrt",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "square",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    "all",
    "amax",
    "any",
    "argmax",
    "argmin",
    "max",
    "mean",
    "min",
    "prod",
    "product",
    "ptp",
    "sometrue",
    "std",
    "sum",
    "var",
    "cumsum",
    "cumprod",
    "cumproduct",
    "sort",
    "scan",
    "fold",
    "vmap",
    "trace",
    "where",
    "clip",
    "tril",
    "triu",
    "add",
    "arctan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "divide",
    "divmod",
    "equal",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "greater",
    "greater_equal",
    "hypot",
    "left_shift",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "nextafter",
    "not_equal",
    "power",
    "remainder",
    "right_shift",
    "subtract",
    "true_divide",
    "auto_sharded",
    "axis_mapping",
    "named_jit",
    "shard_with_axis_mapping",
    "named_jit",
    "enable_shape_checks",
    "are_shape_checks_enabled",
    "isclose",
    "pad_left",
    "stack",
    "concatenate",
]
