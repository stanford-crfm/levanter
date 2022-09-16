import functools
from functools import partial
from typing import Callable, Tuple, TypeVar

import jax
import jax.nn as jnn
import jax.numpy as jnp

from haliax.util import named_call
from levanter.jax_utils import reduce


def quick_gelu(x):
    return x * jnn.sigmoid(1.702 * x)


ACT2FN = {
    "gelu": partial(jnn.gelu, approximate=False),
    "relu": jnn.relu,
    "silu": jnn.silu,
    "swish": jnn.swish,
    "gelu_new": partial(jnn.gelu, approximate=True),
    "gelu_new_remat": jax.remat(partial(jnn.gelu, approximate=True)),
    "quick_gelu": quick_gelu,
}


class RunningMean(object):
    """Numerically stable running mean for an arbitrary array"""

    def __init__(self, shape=(), dtype=jnp.float32):
        self.mean = jnp.zeros(shape, dtype)
        self.count = 0

    def update(self, x):
        self.count += 1
        self.mean += (x - self.mean) / self.count


M = TypeVar("M")
X = TypeVar("X")


# TODO: running mean?
@named_call
def accumulate_gradients(f: Callable[[M, X], Tuple[float, M]], model: M, *inputs: X) -> Tuple[float, M]:
    zero = (jnp.zeros(()), jax.tree_util.tree_map(lambda m: jnp.zeros_like(m), model), 0)

    def compute_and_accumulate(acc, *input):
        loss, grad = f(model, *input)
        acc_loss, acc_grad, n = acc
        return loss + acc_loss, jax.tree_map(jnp.add, acc_grad, grad), n + 1

    total_loss, total_grad, total_n = reduce(compute_and_accumulate, zero, *inputs)

    return total_loss / total_n, jax.tree_map(lambda x: x / total_n, total_grad)


# from https://github.com/google/jax/issues/4285
def recursive_checkpoint(funs, threshold=2):
    if len(funs) == 1:
        return funs[0]
    elif len(funs) == 2:
        f1, f2 = funs
        return lambda x: f2(f1(x))
    elif len(funs) <= threshold:
        return functools.reduce(lambda f, g: lambda x: g(f(x)), funs)
    else:
        f1 = recursive_checkpoint(funs[: len(funs) // 2])
        f2 = recursive_checkpoint(funs[len(funs) // 2 :])
        return lambda x: f2(jax.remat(f1)(x))
