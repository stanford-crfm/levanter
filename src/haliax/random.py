"""Wrappers around jax.random functions."""
import functools
import inspect
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax
from haliax.core import NamedArray, NamedOrNumeric, broadcast_to, selects_axis
from haliax.util import ensure_tuple

from .jax_utils import named_call
from .partitioning import auto_sharded, physical_axis_name, physical_axis_size, pspec_for_axis
from .types import Axis, AxisSelector, AxisSpec


@named_call
def uniform(
    key, shape: AxisSpec, dtype=jnp.float_, minval: NamedOrNumeric = 0.0, maxval: NamedOrNumeric = 1.0
) -> NamedArray:
    shape = ensure_tuple(shape)
    minval = broadcast_to(minval, shape).array
    maxval = broadcast_to(maxval, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.uniform(key=key, shape=jax_shape, dtype=dtype, minval=minval, maxval=maxval)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def normal(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.normal(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def bernoulli(key, shape: AxisSpec, p: NamedOrNumeric):
    shape = ensure_tuple(shape)
    p = broadcast_to(p, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.bernoulli(key=key, p=p, shape=jax_shape)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def randint(key, shape: AxisSpec, minval: NamedOrNumeric, maxval: NamedOrNumeric, dtype=jnp.int_):
    shape = ensure_tuple(shape)
    minval = broadcast_to(minval, shape).array
    maxval = broadcast_to(maxval, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.randint(key=key, shape=jax_shape, minval=minval, maxval=maxval, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def poisson(key, shape: AxisSpec, lam: NamedOrNumeric, dtype=jnp.int_):
    shape = ensure_tuple(shape)
    lam = broadcast_to(lam, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.poisson(key=key, lam=lam, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def exponential(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.exponential(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def gamma(key, shape: AxisSpec, a: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    a = broadcast_to(a, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.gamma(key=key, a=a, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def beta(key, shape: AxisSpec, a: NamedOrNumeric, b: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    a = broadcast_to(a, shape).array
    b = broadcast_to(b, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.beta(key=key, a=a, b=b, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def laplace(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.laplace(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def cauchy(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.cauchy(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def logistic(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.logistic(key=key, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def truncated_normal(key, shape: AxisSpec, lower: NamedOrNumeric, upper: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    lower = broadcast_to(lower, shape).array
    upper = broadcast_to(upper, shape).array
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.truncated_normal(key=key, lower=lower, upper=upper, shape=jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


_enforce_sharded_generate = False
""" mostly for testing: enforces shard generation for all random functions even if not running distributed"""


def generate_sharded(fn, axis: Optional[AxisSelector] = None):
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
        _axis = axis
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        key = bound.arguments["key"]
        shape = bound.arguments["shape"]

        shape = ensure_tuple(shape)

        if len(shape) == 0:
            # scalar
            return fn(*args, **kwargs)

        if _axis is None:
            pspec = pspec_for_axis(shape)
            if pspec:
                _axis, biggest_physical = max(
                    zip(shape, pspec), key=lambda x: (physical_axis_size(x[0]) or 0) if x[1] else 0
                )
            else:
                _axis = biggest_physical = None

            _axis = _axis or shape[0]
        else:
            biggest_physical = physical_axis_name(axis)

        if _enforce_sharded_generate or biggest_physical:
            with jax.named_scope(f"generate_sharded({_axis})"):
                index_of_axis_to_shard = shape.index(_axis)
                # remove axis from shape
                shape_without_axis = shape[:index_of_axis_to_shard] + shape[index_of_axis_to_shard + 1 :]

                keys = jrandom.split(key, _axis.size)

                bound.arguments["shape"] = shape_without_axis
                bound.arguments["key"] = keys

                return haliax.vmap(fn, axis=_axis)(*bound.args, **bound.kwargs).rearrange(shape)
        else:
            with jax.named_scope(f"generate_sharded({_axis}, no_shard)"):
                return fn(*args, **kwargs)

    return wrapped_fn


@named_call
def ball(key, shape: AxisSpec, D: Axis, p: float = 2.0, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.ball(key=key, shape=jax_shape, d=D.size, p=p, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape + (D,)))


@named_call
def choice(
    key, shape: AxisSpec, a: NamedArray, axis: AxisSelector, replace: bool = True, p: Optional[NamedArray] = None
):
    """
    Selects random elements from an array along the given axis. If p is provided, the elements are selected
    with probability proportional to their weights and it must be a 1-d array with its only axis being the axis.
    shape and a.axes must not overlap except that axis may be repeated in both.

    :return: Array with shape `shape` + (`a.axes` - `axis`)
    """

    index = a._lookup_indices(axis)
    assert index is not None, f"axis {axis} not in a"

    shape = ensure_tuple(shape)
    if p is not None:
        assert p.resolve_axis(ensure_tuple(axis)) == p.axes, f"p must be 1D with axis {axis} or be None"

    jax_shape = _to_jax_shape(shape)
    jax_p = p.array if p is not None else None

    jax_array = jrandom.choice(key, a.array, jax_shape, replace=replace, p=jax_p, axis=index)

    expected_shape = shape + tuple(a.axes[:index] + a.axes[index + 1 :])

    return auto_sharded(NamedArray(jax_array, expected_shape))


@named_call
def categorical(key, shape: AxisSpec, logits: NamedArray, axis: AxisSelector):
    """Sample random values from categorical distributions.

    Args:
      key: a PRNG key used as the random key.
      shape: A tuple of nonnegative integers representing the result shape.
        Must be broadcast-compatible with logits without the axis
      logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
        so that `softmax(logits, axis)` gives the corresponding probabilities.
      axis: Axis along which logits belong to the same categorical distribution.
    Returns:
      A random array with int dtype and shape given by ``shape``
    """
    shape = ensure_tuple(shape)

    # TODO: could alias the axis and rename at end
    if selects_axis(shape, axis):
        raise ValueError(f"axis {axis} cannot be in shape {shape}")

    logits = logits.broadcast_axis(shape)

    index = logits._lookup_indices(axis)
    assert index is not None, f"axis {axis} not in logits"

    jax_shape = _to_jax_shape(shape)

    jax_array = jrandom.categorical(key, logits.array, axis=index, shape=jax_shape)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def gumbel(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.gumbel(key, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def permutation(key, x: NamedArray, axis: AxisSelector, independent: bool = False):
    axis_index = x._lookup_indices(axis)
    jax_array = jrandom.permutation(key, x.array, axis_index, independent=independent)
    return auto_sharded(NamedArray(jax_array, x.axes))


@named_call
def rademacher(key, shape: AxisSpec, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.rademacher(key, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def t(key, shape: AxisSpec, df: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    df = broadcast_to(df, shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.t(key, df.array, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def weibull_min(key, shape: AxisSpec, scale: NamedOrNumeric, concentration: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    scale = broadcast_to(scale, shape)
    concentration = broadcast_to(concentration, shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.weibull_min(key, scale.array, concentration.array, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def pareto(key, shape: AxisSpec, b: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    b = broadcast_to(b, shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.pareto(key, b.array, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


@named_call
def loggamma(key, shape: AxisSpec, a: NamedOrNumeric, dtype=jnp.float_):
    shape = ensure_tuple(shape)
    a = broadcast_to(a, shape)
    jax_shape = _to_jax_shape(shape)
    jax_array = jrandom.loggamma(key, a.array, jax_shape, dtype=dtype)
    return auto_sharded(NamedArray(jax_array, shape))


def _to_jax_shape(shape):
    return tuple(axis.size if isinstance(axis, Axis) else axis for axis in shape)


__all__ = [
    "generate_sharded",
    "uniform",
    "normal",
    "ball",
    "bernoulli",
    "beta",
    "cauchy",
    "choice",
    "exponential",
    "gamma",
    "gumbel",
    "laplace",
    "logistic",
    "permutation",
    "poisson",
    "rademacher",
    "truncated_normal",
    # "categorical",
    # "dirichlet",
    "loggamma",
    "pareto",
    "t",
    "weibull_min",
]
