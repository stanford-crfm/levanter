"""Wrappers around jax.random functions."""

import functools
import inspect
from typing import Sequence

import jax.random as jrandom

# TODO: handle broadcasting of array args to random functions (e.g. minval and maxval for uniform)
from haliax.core import Axis, NamedArray
from haliax.util import ensure_tuple, named_call


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
            sig.arguments["shape"] = shape

        # now invoke
        out = func(**sig.arguments)
        if is_haliax:
            return NamedArray(out, orig_shape)

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
