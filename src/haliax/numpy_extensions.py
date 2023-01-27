from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import vmap


# based on from https://github.com/google/jax/issues/3171#issuecomment-1140299630
@partial(jax.jit, static_argnums=(1, 2))
def moving_window(a, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    if axis is None:
        axis = 0
    starts = jnp.arange(a.shape[axis] - size + 1)

    # TODO: np will move the window axis to the end, while this puts it next to the axis it was applied to
    # We could move the axes around to match np, but I'm not sure if that's a good idea for perf???
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
