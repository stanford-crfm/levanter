import time

import jax
import numpy as np
from jax.sharding import Mesh

import haliax as hax
from haliax.partitioning import named_jit

from levanter.models.longformer import causal_sliding_window_attention


Len = hax.Axis("Len", 8192)
W = hax.Axis("W", 512)
D = hax.Axis("D", 4096)
B = hax.Axis("B", 256)

# Len = hax.Axis("Len", 64)
# W = hax.Axis("W", 4)
# D = hax.Axis("D", 8)
# B = hax.Axis("B", 4)


devices = np.array(jax.devices())
mesh = Mesh(devices, ("data",))

axis_resources = {"B": "data"}


@named_jit(axis_resources=axis_resources)
def do_attn(inputs):
    return causal_sliding_window_attention(Len, W, D, inputs, inputs, inputs)


@named_jit(axis_resources=axis_resources)
def init():
    return hax.random.uniform(jax.random.PRNGKey(0), (B, Len, D))


if __name__ == "__main__":
    with mesh:
        data = init()
        result = do_attn(data)
        result.array.block_until_ready()
        time_in = time.time()
        result2 = do_attn(data)
        result2.array.block_until_ready()
        time_out = time.time()
        print(time_out - time_in)
