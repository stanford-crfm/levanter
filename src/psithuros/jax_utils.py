from typing import Union, Tuple

import jax
import numpy as np
from jax import random as jrandom, numpy as jnp


def shaped_rng_split(key, split_shape: Union[int, Tuple[int, ...]] = 2) -> jrandom.KeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits, -1)
    else:
        num_splits = np.prod(split_shape)
        split_shape = split_shape + (-1,)

    if num_splits == 1:
        return jnp.reshape(key, split_shape)

    unshaped = jrandom.split(key, num_splits)
    return jnp.reshape(unshaped, split_shape)


def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.
    Args:
      tree: a pytree containing the arrays that should be replicated.
      devices: the devices the data is replicated to
        (default: same order as expected by `jax.pmap()`).
    Returns:
      A new pytree containing the replicated arrays.
    """
    return jax.device_put_replicated(tree, devices or jax.devices())


def jnp_to_python(a: jnp.ndarray):
    if a.shape == () or a.shape == (1,):
        return a.item()
    else:
        return a.tolist()
