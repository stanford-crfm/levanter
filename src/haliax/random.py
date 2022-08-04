"""Wrappers around jax.random functions."""

import jax.random as jrandom

import functools

from haliax import _ensure_sequence
from haliax.core import *



# TODO: handle broadcasting of array args to random functions (e.g. minval and maxval for uniform)
def _wrap_random_function(func):
    """Wrap a jax random function to return a NamedArray and takes axes as inputs"""
    @functools.wraps(func)
    def wrapper(key: jrandom.KeyArray, shape: AxisSpec, *args, **kwargs):
        shape = _ensure_sequence(shape)
        lens = [ax.size for ax in shape]

        return NamedArray(func(key, lens, *args, **kwargs), shape)
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
    'uniform',
    'normal',
    'randint',
    'bernoulli',
    'poisson',
    'exponential',
    'gamma',
    'beta',
    'laplace',
    'cauchy',
    'logistic',
    'truncated_normal',
]