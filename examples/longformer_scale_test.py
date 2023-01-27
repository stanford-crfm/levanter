import jax
import numpy as np
from jax.experimental.maps import Mesh

import haliax as hax
from haliax.partitioning import named_pjit
from levanter.models.longformer import causal_sliding_window_attention


Len = hax.Axis("Len", 8192)
W = hax.Axis("W", 512)
D = hax.Axis("D", 4096)
B = hax.Axis("B", 256)

devices = np.array(jax.devices())
mesh = Mesh(devices, ("data",))

axis_resources = {"B": "data"}


@named_pjit(axis_resources=axis_resources)
def do_attn(inputs):
    result = causal_sliding_window_attention(Len, W, D, inputs, inputs, inputs)

    return result


data = hax.random.uniform(jax.random.PRNGKey(0), (B, Len, D))


if __name__ == "__main__":
    with mesh:
        result = do_attn(data)
