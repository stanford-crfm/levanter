import jax
import jax.numpy as jnp
from jax import vmap


# from https://github.com/google/jax/issues/3171#issuecomment-1140299630
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)
