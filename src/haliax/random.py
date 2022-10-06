"""Wrappers around jax.random functions."""

import functools
import inspect
from typing import Sequence

import jax
import jax.random as jrandom

# TODO: handle broadcasting of array args to random functions (e.g. minval and maxval for uniform)
from haliax.core import Axis, NamedArray
from haliax.util import ensure_tuple, named_call

from .partitioning import pspec_for_axis, auto_sharded, physical_axis_size


def _wrap_random_function(func):
    """Wrap a jax random function to return a NamedArray and takes axes as inputs"""

    @named_call(name=func.__name__)
    @functools.wraps(func)
    def wrapper(key: jrandom.KeyArray, *args, **kwargs):
        sig: inspect.BoundArguments = inspect.signature(func).bind(key, *args, **kwargs)
        sig.apply_defaults()

        # get shape
        orig_shape = sig.arguments["shape"]
        orig_shape = ensure_tuple(orig_shape)
        if isinstance(orig_shape, Sequence):
            is_haliax = len(orig_shape) == 0 or any(isinstance(s, Axis) for s in orig_shape)
            shape = tuple(axis.size if isinstance(axis, Axis) else axis for axis in orig_shape)
        else:
            is_haliax = False

        if is_haliax:

            # this is a bit tricky but, for sharded models, we sometimes want to split the random key so that we only
            # need to generate the random numbers for the local shard. We do this because the RNG can't actually
            # auto-shard, meaning that if you want to generate a [1024] vector across 4 devices, each one actually
            # generates all 1024 numbers, and then only uses 256 of them. This is a waste of time, especially when it's
            # not a [1024] vector but a [1600, 6400] matrix (for say, gpt-2). So we split the key here, and then let
            # vmap hopefully only generate the random numbers for the local shard.
            #
            # However, we don't want to oversplit or it kind of ruins the whole point since we have to split the key on
            # every node... So instead we just split along the *largest* physical axis
            # TODO: we won't need to do this when they add better splitting for random numbers
            #  (froystig is maybe going to do this?)

            # what we do is we take the biggest axis that is sharded and split on it, ties going to the first axis
            pspec = pspec_for_axis(orig_shape)
            if pspec:
                biggest_axis, biggest_physical = max(zip(orig_shape, pspec), key=lambda x: (physical_axis_size(x[0]) or 0) if x[1] else 0)
            else:
                biggest_axis = biggest_physical = None

            if biggest_physical and biggest_axis.size > 1:
                index_of_biggest_axis = orig_shape.index(biggest_axis)
                shape = shape[:index_of_biggest_axis] + shape[index_of_biggest_axis + 1 :]
                sig.arguments["shape"] = shape
                keys = jrandom.split(key, biggest_axis.size)

                def fn(key):
                    return func(key, *sig.args[1:], **sig.kwargs)

                out = jax.vmap(fn, in_axes=(0,), out_axes=index_of_biggest_axis)(keys)
                return NamedArray(out, orig_shape)
            else:
                sig.arguments["shape"] = shape
                out = func(**sig.arguments)
                return NamedArray(out, orig_shape)
        else:
            return func(**sig.arguments)

    return wrapper


uniform = _wrap_random_function(jrandom.uniform)
normal = _wrap_random_function(jrandom.normal)
randint = _wrap_random_function(jrandom.randint)
bernoulli = _wrap_random_function(jrandom.bernoulli)
poisson = _wrap_random_function(jrandom.poisson)
exponential = _wrap_random_function(jrandom.exponential)
gamma = _wrap_random_function(jrandom.gamma)
beta = _wrap_random_function(jrandom.beta)
laplace = _wrap_random_function(jrandom.laplace)
cauchy = _wrap_random_function(jrandom.cauchy)
logistic = _wrap_random_function(jrandom.logistic)
truncated_normal = _wrap_random_function(jrandom.truncated_normal)

__all__ = [
    "uniform",
    "normal",
    "randint",
    "bernoulli",
    "poisson",
    "exponential",
    "gamma",
    "beta",
    "laplace",
    "cauchy",
    "logistic",
    "truncated_normal",
]
