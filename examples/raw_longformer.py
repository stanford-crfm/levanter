from functools import partial

import jax
import jax.numpy as jnp

from typing import Optional

from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit, PartitionSpec as P

mesh = Mesh(jax.devices(), ("data",))

# based on from https://github.com/google/jax/issues/3171#issuecomment-1140299630
@partial(jax.jit, static_argnums=(1, 2))
def moving_window1(a, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    if axis is None:
        axis = 0
    starts = jnp.arange(a.shape[axis] - size + 1)

    return jax.vmap(lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis=axis), out_axes=0)(starts)


# inspired by https://github.com/google/jax/issues/1646#issuecomment-1139044324
@partial(jax.jit, static_argnums=(1, 2))
def moving_window2(a: jnp.ndarray, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    """ It also returns the axes in a different order: the returned shape is
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


@partial(jax.jit, static_argnums=(1, 2))
def moving_window3(a, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    if axis is None:
        axis = 0
    starts = jnp.arange(a.shape[axis] - size + 1).reshape((-1, 1))
    indices = jnp.arange(size).reshape((1, -1))
    slices = starts + indices
    return jnp.take(a, slices, axis=axis)


@partial(jax.jit, static_argnums=(1, 2))
def moving_window4(a, size: int, axis: Optional[int] = None) -> jnp.ndarray:
    axis = axis or 0
    axis_size = a.shape[axis]
    num_windows = axis_size - size + 1
    tiles = [1] * a.ndim
    tiles[axis] = num_windows + 1
    tiled = jnp.tile(a, tiles)

    # slide one element along the axis
    sliced = jax.lax.dynamic_slice_in_dim(tiled, 0, num_windows * (axis_size + 1), axis=axis)
    new_shape = a.shape[:axis] + (num_windows, axis_size + 1) + a.shape[axis + 1:]
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

    padded = pad(a, padding, size - 1, axis)
    return moving_window4(padded, size, axis=axis)

def pad(a, padding, size, axis):
    padding_config = [(0, 0, 0)] * a.ndim
    padding_config[axis] = (size, 0, 0)
    padded = jax.lax.pad(a, padding_value=padding, padding_config=padding_config)
    return padded


B = len(jax.devices())
L = 8192
W = 512
D = 4096
# L = 64
# W = 4
# D = 8

def fake_attn(query, key):
    # padded_key = padded_moving_window(key, W, -1000.0, axis=1)
    # return jnp.einsum("bld,lbwd->blw", query, padded_key)




def vmap_variant1(query, key):
    def do_one_pos(q, k, l):
        k = jax.lax.dynamic_slice_in_dim(k, l, W, axis=1)
        return jnp.einsum("bd,bwd->bw", q, k)

    padded_k = pad(key, -1000.0, W, axis=1)
    return jax.vmap(do_one_pos, in_axes=(1, None, 0), out_axes=1)(query, padded_k, jnp.arange(L))


def vmap_variant2(query, key):
    def do_one_pos(q, k, l):
        k = jnp.roll(k, -l, axis=1)
        k = jax.lax.dynamic_slice_in_dim(k, 0, W, axis=1)
        return jnp.einsum("bd,bwd->bw", q, k)

    return jax.vmap(do_one_pos, in_axes=(1, None, 0), out_axes=1)(query, key, jnp.arange(L))


def init():
    return jnp.ones((B, L, D))


pjit_fake_attn = pjit(fake_attn, in_axis_resources=P("data"), out_axis_resources=P("data"))
pjit_init = pjit(init, out_axis_resources=P("data"))
pjit_vmap_variant1 = pjit(vmap_variant1, in_axis_resources=P("data"), out_axis_resources=P("data"))

if __name__ == "__main__":
    with mesh:
        data = pjit_init()
        # result = pjit_fake_attn(data, data)
        result = pjit_vmap_variant1(data, data)
