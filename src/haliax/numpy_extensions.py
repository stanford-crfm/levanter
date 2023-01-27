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
    if axis is None:
        axis = 0
    starts = jnp.arange(a.shape[axis] - size + 1)

    def roll(x, i):
        return jnp.roll(x, -i, axis=axis)

    rolled = jax.vmap(roll, in_axes=(None, 0), out_axes=0)(a, starts)
    # rolled has shape ( a.shape[axis] - size + 1, *a.shape
    # but we want (a.shape[axis] - size + 1, a.shape[:axis], size, a.shape[axis+1:])
    # so we need to slice out the axis we want
    desired_shape = (a.shape[axis] - size + 1, *a.shape[:axis], size, *a.shape[axis + 1 :])
    return jax.lax.slice(rolled, (0,) * rolled.ndim, desired_shape)

    # TODO: maybe better to just leave out_axes = 0

    # TODO: np will move the window axis to the end, while this puts it next to the axis it was applied to
    # We could move the axes around to match np, but I'm not sure if that's a good idea for perf???
    # return vmap(lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis=axis), out_axes=axis)(starts)
    # num_windows = a.shape[axis] - size + 1

    # # the way we're going to do this is by broadcasting the array num_windows times, then ravel, then reshape
    # # [a b c d e] W=3
    # broadcasted = jax.lax.broadcast_in_dim(a, (num_windows, *a.shape), tuple(range(1, a.ndim)))
    # # [[a b c d e] [a b c d e] [a b c d e] ...]
    # raveled = broadcasted.ravel()
    # # [a b c d e a b c d e a b c d e ...]
    # # now we want
    # # [[a b c] [b c d] [c d e] ...]
    # # to do that we first reshape to
    # # [[a b c d e a] [b c d e a b] [c d e a b c] ...]
    # # then we can slice out the first 3 elements of each row
    # reshaped = raveled.reshape((num_windows, ))


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
