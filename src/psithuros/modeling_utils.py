from functools import partial
from typing import Callable, Any, Optional, TypeVar

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
