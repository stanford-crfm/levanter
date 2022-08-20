import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import Mesh, PartitionSpec


devices = jax.devices()
mesh = Mesh(np.array(devices).reshape((-1, 4)), ["data", "model"])

with mesh:
    spec = PartitionSpec("data")
    array = jnp.zeros(512, dtype=jnp.bfloat16)

    def foo(weight, data):
        return weight * data

    def foo_novmap(weight, data):
        return weight.reshape(1, -1) * data

    # ok
    out1 = foo_novmap(array, jnp.zeros((256, 512)))
    print(out1.shape)

    # error
    pjit_foo_novmap = pjit(foo_novmap, in_axis_resources=(spec, PartitionSpec("data", None)), out_axis_resources=spec)
    out = pjit_foo_novmap(array, jnp.zeros((256, 512)))

    # ok
    vmap_foo = jax.vmap(foo, in_axes=(None, 0))
    out1 = vmap_foo(array, jnp.zeros((256, 512)))
    print(out1.shape)

    # also error
    pjit_foo = pjit(vmap_foo, in_axis_resources=(spec, PartitionSpec("data", None)), out_axis_resources=spec)
    out = pjit_foo(array, jnp.zeros((256, 512)))
    print(out.shape)
