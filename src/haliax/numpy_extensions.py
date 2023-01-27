from typing import Optional

import jax
import jax.numpy as jnp
from jax import vmap


# based on from https://github.com/google/jax/issues/3171#issuecomment-1140299630
def moving_window(a, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    if axis is None:
        axis = 0
    starts = jnp.arange(a.shape[axis] - size + 1)

    # TODO: if we set out_axes=-1 we get something closer to numpy behavior. see if it matters for xla/jit
    return vmap(lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis=axis), out_axes=axis)(starts)


def padded_moving_window(
    a,
    size: int,
    padding,
    axis: Optional[int] = None,
) -> jnp.ndarray:
    """Similar to moving_window, but pads the array with the given padding value before applying the window so that
    the output has the same shape as the input (plus the window)"""
    if axis is None:
        axis = 0

    padding_config = [(0, 0, 0)] * a.ndim
    padding_config[axis] = (size - 1, 0, 0)

    padded = jax.lax.pad(a, padding_value=padding, padding_config=padding_config)
    return moving_window(padded, size, axis=axis)
