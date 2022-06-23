from typing import Union, Tuple, Optional, Callable, TypeVar, Sequence

import jax
import numpy as np
from chex import PRNGKey
from jax import random as jrandom, numpy as jnp, lax


def maybe_rng_split(key: Optional[PRNGKey], num: int = 2):
    """Splits a random key into multiple random keys. If the key is None, then it replicates the None. Also handles
    num == 1 case """
    if key is None:
        return [None] * num
    elif num == 1:
        return jnp.stack(key)
    else:
        return jrandom.split(key, num)


def shaped_rng_split(key, split_shape: Union[int, Sequence[int]] = 2) -> jrandom.KeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits, ) + key.shape
    else:
        num_splits = np.prod(split_shape)
        split_shape = tuple(split_shape) + key.shape

    if num_splits == 1:
        return jnp.reshape(key, split_shape)

    unshaped = maybe_rng_split(key, num_splits)
    return jnp.reshape(unshaped, split_shape)


def jnp_to_python(a: jnp.ndarray):
    if a.shape == () or a.shape == (1,):
        return a.item()
    else:
        return a.tolist()

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')

def fold_left(fn: Callable[[Carry, X], Carry], init: Carry, *xs: X) -> Carry:
    # lax.scan that calls an xmapped function seems to break jax.
    # res = lax.scan(lambda carry, x: (fn(carry, x), None), init=init, xs=xs)
    # return res[0]
    ys = zip(*xs)
    for x in ys:
        init = fn(init, x)
    return init


def flops_estimate(fn, *args):
    """Estimates the flop count of a function using XLA/HLO fanciness. See https://github.com/google/flax/discussions/1854"""
    m = jax.xla_computation(fn)(*args).as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    costs = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)
    return costs['flops']


def dump_jaxpr(file, fn, *args, **kwargs):
    jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    with open(file, "w") as f:
        f.write(jaxpr.pretty_print(source_info=True, name_stack=True))