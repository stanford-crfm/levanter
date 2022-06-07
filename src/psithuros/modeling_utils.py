from functools import partial

import jax.nn as jnn
import jax.numpy as jnp




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
