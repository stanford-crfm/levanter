from typing import Union, Tuple, Optional, Callable, TypeVar

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


def shaped_rng_split(key, split_shape: Union[int, Tuple[int, ...]] = 2) -> jrandom.KeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits, -1)
    else:
        num_splits = np.prod(split_shape)
        split_shape = split_shape + (-1,)

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

def fold_left(fn: Callable[[Carry, X], Carry], init: Carry, xs: X) -> Carry:
    res = lax.scan(lambda carry, x: (fn(carry, x), None), init=init, xs=xs)
    return res[0]