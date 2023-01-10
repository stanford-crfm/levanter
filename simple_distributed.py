import jax
import jax.numpy as jnp
import transformers
from jax.experimental import multihost_utils

jax.distributed.initialize()

print(jax.devices())
print(jax.process_count())

buf = jnp.zeros(100)

# The psum is performed over all mapped devices across the pod slice
xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)

print(r)

# out = multihost_utils.broadcast_one_to_all(buf)
#
# print(out)
