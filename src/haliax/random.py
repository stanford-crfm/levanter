"""Wrappers around jax.random functions."""
import functools
import inspect
from typing import Optional

import jax.numpy as jnp
import jax.random as jrandom

import haliax
from haliax.core import NamedArray, NamedOrNumeric, broadcast_to
from haliax.util import ensure_tuple

from .jax_utils import named_call
from .partitioning import auto_sharded, physical_axis_name, physical_axis_size, pspec_for_axis
from .types import Axis, AxisSpec


@named_call
def uniform(
    key, shape: AxisSpec, dtype=jnp.float_, minval: NamedOrNumeric = 0.0, maxval: NamedOrNumeric = 1.0
) -> NamedArray:
    shape = ensure_tuple(shape)
    minval = broadcast_to(minval, shape).array
    maxval = broadcast_to(maxval, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.uniform(key=key, shape=jax_shape, dtype=dtype, minval=minval, maxval=maxval)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def normal(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.normal(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def bernoulli(key, p: NamedOrNumeric, shape: AxisSpec):
    shape = ensure_tuple(shape)
    p = broadcast_to(p, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.bernoulli(key=key, p=p, shape=jax_shape)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def randint(key, shape: AxisSpec, minval: NamedOrNumeric, maxval: NamedOrNumeric, dtype=jnp.int_):
    shape = ensure_tuple(shape)
    minval = broadcast_to(minval, shape).array
    maxval = broadcast_to(maxval, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.randint(key=key, shape=jax_shape, minval=minval, maxval=maxval, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def poisson(key, lam: NamedOrNumeric, shape: AxisSpec, dtype=jnp.int_):
    shape = ensure_tuple(shape)
    lam = broadcast_to(lam, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.poisson(key=key, lam=lam, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def exponential(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.exponential(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def gamma(key, a: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    a = broadcast_to(a, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.gamma(key=key, a=a, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def beta(key, a: NamedOrNumeric, b: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    a = broadcast_to(a, shape).array
    b = broadcast_to(b, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.beta(key=key, a=a, b=b, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def laplace(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.laplace(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def cauchy(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.cauchy(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def logistic(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.logistic(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def truncated_normal(key, lower: NamedOrNumeric, upper: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    lower = broadcast_to(lower, shape).array
    upper = broadcast_to(upper, shape).array
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.truncated_normal(key=key, lower=lower, upper=upper, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


_enforce_sharded_generate = False
""" mostly for testing: enforces shard generation for all random functions even if not running distributed"""


def generate_sharded(fn, axis: Optional[Axis] = None):
    """
    Create a wrapped version of fn (which should be a random generator) that generates the random array in a sharded
    manner, using vmap over the provided axis, or inferring the "best" one if not provided.

    This is a bit tricky but, for sharded models, we sometimes want to split the random key so that we only
    need to generate the random numbers for the local shard. We do this because the RNG can't actually
    auto-shard, meaning that if you want to generate a [1024] vector across 4 devices, each one actually
    generates all 1024 numbers, and then only uses 256 of them. This is a waste of time, especially when it's
    not a [1024] vector but a [1600, 6400] matrix (for say, gpt-2). So we split the key here, and then let
    vmap hopefully only generate the random numbers for the local shard.



    However, we don't want to oversplit or it kind of ruins the whole point since we have to split the key on
    every node... So instead we just split along the *largest* physical axis, or the provided axis if it's
    provided.
    """
    # TODO: we won't need to do this when they add better splitting for random numbers
    #  (froystig is maybe going to do this?)

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        key = bound.arguments["key"]
        shape = bound.arguments["shape"]

        shape = ensure_tuple(shape)

        if len(shape) == 0:
            # scalar
            return fn(*args, **kwargs)

        if axis is None:
            pspec = pspec_for_axis(shape)
            if pspec:
                axis_to_shard, biggest_physical = max(
                    zip(shape, pspec), key=lambda x: (physical_axis_size(x[0]) or 0) if x[1] else 0
                )
            else:
                axis_to_shard = biggest_physical = None

            axis_to_shard = axis_to_shard or shape[0]
        else:
            axis_to_shard = axis
            biggest_physical = physical_axis_name(axis)

        if _enforce_sharded_generate or biggest_physical:
            index_of_axis_to_shard = shape.index(axis_to_shard)
            # remove axis from shape
            shape = shape[:index_of_axis_to_shard] + shape[index_of_axis_to_shard + 1 :]

            keys = jrandom.split(key, axis_to_shard.size)

            bound.arguments["shape"] = shape
            bound.arguments["key"] = keys

            return haliax.vmap(lambda *args, **kwargs: fn(*args, **kwargs), axis=axis)(*bound.args, **bound.kwargs)
        else:
            return fn(*args, **kwargs)

    return wrapped_fn


def ball(key, shape: AxisSpec, d: int, p: float = 2.0, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.ball(key=key, shape=jax_shape, d=d, p=p, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


# TODO: categorical
# def categorical(key, logits: NamedArray, axis: Axis, shape: Optional[AxisSpec] = None):
#     # TODO: this one is tricky. need to test carefully
#     if shape is not None:
#         logits = logits.broadcast_axis(shape)
#     else:
#         shape = logits.axes
#         # TODO: add delete_axis
#         index_of_axis = shape.index(axis)
#         shape = shape[:index_of_axis] + shape[index_of_axis + 1 :]
#
#     axis_index = logits._lookup_indices(axis)
#
#     jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
#
#     return auto_sharded(NamedArray(jrandom.categorical(key, logits.array, axis_index, jax_shape), shape))


def choice(key, a: NamedArray, shape: AxisSpec, replace: bool = True, p: Optional[NamedArray] = None):
    shape = ensure_tuple(shape)
    if p is not None:
        p = p.broadcast_axis(shape)
    a = a.broadcast_axis(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.choice(key, a.array, jax_shape, replace=replace, p=p.array if p is not None else None)
    return auto_sharded(NamedArray(jax_array, shape))


# TODO: dirichlet
#     A random array with the specified dtype and shape given by
#     ``shape + (alpha.shape[-1],)`` if ``shape`` is not None, or else
#     ``alpha.shape``.
# def dirichlet(key, alpha: NamedArray, shape: Optional[AxisSpec] = None):
#     if shape is not None:
#         alpha = alpha.broadcast_axis(shape)
#     jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape) if shape is not None else None
#     jax_array = jrandom.dirichlet(key, alpha.array, jax_shape)
#     return auto_sharded(NamedArray(jax_array, alpha.axes))


def gumbel(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.gumbel(key, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


# def loggamma(key, a: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
#     shape = ensure_tuple(shape)
#     a = broadcast_to(a, shape)
#     jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
#     jax_array = jrandom.loggamma(key, a, jax_shape, dtype=dtype)
#     return auto_sharded(NamedArray(jax_array, shape))


def orthogonal(key, n: int, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.orthogonal(key, n, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


# def pareto(key, b: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
#     shape = ensure_tuple(shape)
#     b = broadcast_to(b, shape)
#     jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
#     jax_array = jrandom.pareto(key, b, jax_shape, dtype=dtype)
#     return auto_sharded(NamedArray(jax_array, shape))


def permutation(key, x: NamedArray, axis: Axis, independent: bool = False):
    axis_index = x._lookup_indices(axis)
    jax_array = jrandom.permutation(key, x.array, axis_index, independent=independent)
    return auto_sharded(NamedArray(jax_array, x.axes))


def rademacher(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
    jax_array = jrandom.rademacher(key, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


# def t(key, df: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
#     shape = ensure_tuple(shape)
#     df = broadcast_to(df, shape)
#     jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
#     jax_array = jrandom.t(key, df, jax_shape, dtype=dtype)
#     return auto_sharded(NamedArray(jax_array, shape))


# def weibull_min(key, scale: NamedOrNumeric, concentration: NamedOrNumeric, shape: AxisSpec, dtype=jnp.float_):
#     shape = ensure_tuple(shape)
#     scale = broadcast_to(scale, shape)
#     concentration = broadcast_to(concentration, shape)
#     jax_shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)
#     jax_array = jrandom.weibull_min(key, scale, concentration, jax_shape, dtype=dtype)
#     return auto_sharded(NamedArray(jax_array, shape))


__all__ = [
    "generate_sharded",
    "uniform",
    "normal",
    "bernoulli",
    "poisson",
    "permutation",
    "choice",
    "laplace",
    "exponential",
    "gamma",
    # "pareto",
    "beta",
    # "dirichlet",
    "gumbel",
    # "loggamma",
    "rademacher",
    # "t",
    # "weibull_min",
    "orthogonal",
    # "categorical",
    "ball",
]
