from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp


# based on from https://github.com/google/jax/issues/3171#issuecomment-1140299630
@partial(jax.jit, static_argnums=(1, 2))
def moving_window(a: jnp.ndarray, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    """Unlike jax.numpy.lib.stride_tricks.sliding_window_view, this function does not necessarily return a view
    of the original array.

    It also returns the axes in a different order: the returned shape is
     `(a.shape[axis] - size + 1, *a.shape[:axis], size, *a.shape[axis + 1 :])`
    that is, the number of windows is the first axis, then the axes before the window axis, then the window axis, then
    the axes after the window axis.
    """
    axis = axis or 0
    axis_size = a.shape[axis]
    num_windows = axis_size - size + 1
    tiles = [1] * a.ndim
    tiles[axis] = num_windows + 1
    tiled = jnp.tile(a, tiles)

    # slide one element along the axis
    sliced = jax.lax.dynamic_slice_in_dim(tiled, 0, num_windows * (axis_size + 1), axis=axis)
    new_shape = a.shape[:axis] + (num_windows, axis_size + 1) + a.shape[axis + 1 :]
    reshaped = jnp.reshape(sliced, new_shape)

    # now slice out the windows
    return jax.lax.dynamic_slice_in_dim(reshaped, 0, size, axis=axis + 1)


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
