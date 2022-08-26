import jax.numpy as jnp

import haliax.random as random

from .core import (
    Axis,
    AxisSpec,
    NamedArray,
    concat_axis_specs,
    dot,
    flatten_axes,
    named,
    rearrange,
    rename,
    split,
    take,
    unbind,
    unflatten_axis,
)
from .hof import fold_left, scan, vmap
from .wrap import wrap_elemwise_unary, wrap_normalization_call, wrap_reduction_call


# creation routines
def zeros(shape: AxisSpec, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 0"""
    return full(shape, 0, dtype)


def ones(shape: AxisSpec, dtype=None) -> NamedArray:
    """Creates a NamedArray with all elements set to 1"""
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


def arange(axis: Axis, dtype=None) -> NamedArray:
    return NamedArray(jnp.arange(axis.size, dtype=dtype), (axis,))


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
all = wrap_reduction_call(jnp.all)
amax = wrap_reduction_call(jnp.amax)
any = wrap_reduction_call(jnp.any)
# argmax = wrap_reduction_call(jnp.argmax)
# argmin = wrap_reduction_call(jnp.argmin)
max = wrap_reduction_call(jnp.max)
mean = wrap_reduction_call(jnp.mean)
min = wrap_reduction_call(jnp.min)
prod = wrap_reduction_call(jnp.prod)
product = wrap_reduction_call(jnp.product)
sometrue = wrap_reduction_call(jnp.sometrue)
std = wrap_reduction_call(jnp.std)
sum = wrap_reduction_call(jnp.sum)
var = wrap_reduction_call(jnp.var)

# "Normalization" functions that use an axis but don't change the shape
cumsum = wrap_normalization_call(jnp.cumsum, True)
cumprod = wrap_normalization_call(jnp.cumprod, True)
cumproduct = wrap_normalization_call(jnp.cumproduct, True)

__all__ = [
    "Axis",
    "NamedArray",
    "AxisSpec",
    "named",
    "dot",
    "split",
    "flatten_axes",
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
    "max",
    "mean",
    "min",
    "prod",
    "product",
    "sometrue",
    "std",
    "sum",
    "var",
    "cumsum",
    "cumprod",
    "cumproduct",
    "scan",
    "fold_left",
    "vmap",
]
