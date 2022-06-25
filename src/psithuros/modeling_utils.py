from functools import partial
from typing import Callable, Tuple, TypeVar

import jax
import jax.nn as jnn
import jax.numpy as jnp

from psithuros.jax_utils import fold_left


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


class RunningMean(object):
    """Numerically stable running mean for an arbitrary array"""

    def __init__(self, shape = (), dtype=jnp.float32):
        self.mean = jnp.zeros(shape, dtype)
        self.count = 0

    def update(self, x):
        self.count += 1
        self.mean += (x - self.mean) / self.count


M = TypeVar('M')
X = TypeVar('X')


# TODO: running mean?
def accumulate_gradients(f: Callable[[M, X], Tuple[float, M]], model: M, *inputs: X) -> Tuple[float, M]:
    zero = (jnp.zeros(()), jax.tree_map(jnp.zeros_like, model), 0)
    def compute_and_accumulate(acc, *input):
        loss, grad = f(model, *input)
        acc_loss, acc_grad, n = acc
        # TODO: verify if we still need this now that we have scan working again
        # in place to preserve ram
        def add(x, y):
            x += y
            return x

        return loss + acc_loss, jax.tree_map(add, acc_grad, grad), n + 1
    total_loss, total_grad, total_n = fold_left(compute_and_accumulate, zero, *inputs)

    return total_loss/total_n, jax.tree_map(lambda x: x/total_n, total_grad)


