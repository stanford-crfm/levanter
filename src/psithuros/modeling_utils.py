from functools import partial
from typing import Callable, Any, Optional, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.nn as jnn

from equinox.custom_types import Array
from jax import lax

Carry = TypeVar('Carry')
X = TypeVar('X')
Y = TypeVar('Y')


def fold_left(fn: Callable[[Carry, X], Carry], init: Carry, xs: X) -> Carry:
    res = lax.scan(lambda carry, x: (fn(carry, x), None), init=init, xs=xs)
    return res[0]


def quick_gelu(x):
    return x * jnn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(jnn.gelu, approximate=False),
    "relu": jnn.relu,
    "silu": jnn.silu,
    "swish": jnn.swish,
    "gelu_new": partial(jnn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}


def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.
    Args:
      tree: a pytree containing the arrays that should be replicated.
      devices: the devices the data is replicated to
        (default: same order as expected by `jax.pmap()`).
    Returns:
      A new pytree containing the replicated arrays.
    """
    return jax.device_put_replicated(tree, devices or jax.devices())
