# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import asyncio
import functools
import logging
import os
from functools import partial
from typing import Callable, Optional

import equinox
import jax
import jax.experimental.array_serialization.serialization as array_ser
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tensorstore
from jax.sharding import Mesh
from tensorstore import TensorStore

import haliax as hax
import haliax.tree_util as htu
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array

from levanter.utils import jax_utils


logger = logging.getLogger(__name__)


def _is_named_or_none(x):
    return x is None or is_named_array(x)


def tree_serialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    manager: Optional[array_ser.GlobalAsyncCheckpointManager] = None,
    *,
    commit_callback: Optional[Callable] = None,
):
    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()
        manager_was_none = True
    else:
        manager_was_none = False

    leaf_key_paths = jax_utils.leaf_key_paths(pytree, is_leaf=_is_named_or_none)

    def path_from_key_path(key_path):
        return os.path.join(checkpoint_dir, *key_path.split("."))

    paths = jtu.tree_map(path_from_key_path, leaf_key_paths, is_leaf=lambda x: x is None)
    paths = jtu.tree_leaves(paths, is_leaf=lambda x: x is None)
    leaves = jtu.tree_leaves(pytree, is_leaf=lambda x: x is None)
    assert len(leaves) == len(paths)

    # ok, not all of these are arrays, but we'll deal with that in the async function
    def _ensure_is_array(x):
        if isinstance(x, (int, float, bool, complex)):
            return jnp.array(x)
        else:
            return x

    arrays = [_ensure_is_array(x) for x in leaves]

    # filter out the None leaves and paths (must be zip)
    arrays, paths = zip(*[(a, p) for a, p in zip(arrays, paths) if equinox.is_array_like(a)])

    arrays = list(arrays)
    paths = list(paths)

    if commit_callback is None:
        commit_callback = lambda: logger.info("Committed checkpoint to Tensorstore")  # noqa

    manager.serialize_with_paths(arrays, paths, on_commit_callback=commit_callback)

    if manager_was_none:
        manager.wait_until_finished()


def _tensorstore_spec_for(checkpoint_dir, key_path: str):
    checkpoint_path = os.path.join(checkpoint_dir, *key_path.split("."))
    ts_spec = array_ser.get_tensorstore_spec(checkpoint_path)
    return ts_spec


async def _serialize_one_leaf(x, spec):
    if isinstance(x, hax.NamedArray):
        # we don't need to do anything special for named arrays to serialize, though we will for deserialization.
        return await _serialize_one_leaf(x.array, spec)
    elif isinstance(x, jax.Array):
        if not x.is_fully_addressable:
            return await array_ser.async_serialize(x, spec)
        else:
            return await save_array_to_tensorstore(x, spec)
    elif isinstance(x, (bool, float, complex, int)):
        return await save_array_to_tensorstore(np.array(x), spec)
    elif x is None:
        return
    elif isinstance(x, jnp.ndarray):
        return await save_array_to_tensorstore(x, spec)
    elif isinstance(x, np.ndarray):
        return await save_array_to_tensorstore(x, spec)
    else:
        raise TypeError(f"Can't serialize {type(x)}")


async def save_array_to_tensorstore(x, spec):
    if jax.process_index() == 0:
        if x.dtype == jnp.bfloat16:
            # Tensorstore uses 'bfloat16', not '<V2'.
            dtype = "bfloat16"
        else:
            dtype = np.dtype(x.dtype).str
        t = await tensorstore.open(
            tensorstore.Spec(spec), create=True, shape=x.shape, dtype=dtype, context=array_ser.TS_CONTEXT
        )

        await t.write(x)


async def load_array_from_tensorstore(spec):
    t: TensorStore = await tensorstore.open(tensorstore.Spec(spec), context=array_ser.TS_CONTEXT)
    return await t.read()


async def _deserialize_one_leaf(like, spec, axis_mapping, mesh):
    if is_named_array(like):
        return await _deserialize_named_array(like, spec, axis_mapping, mesh)
    elif isinstance(like, jax.Array):
        if not like.is_fully_addressable:
            return await array_ser.async_deserialize(like.sharding, spec, global_shape=like.shape, dtype=like.dtype)
        else:
            return await load_array_from_tensorstore(spec)
    elif isinstance(like, (bool, float, complex, int)):
        arr = await load_array_from_tensorstore(spec)
        return arr.item()
    elif like is None:
        return None
    elif isinstance(like, jnp.ndarray) or isinstance(like, np.ndarray) or isinstance(like, jax.ShapeDtypeStruct):
        return await load_array_from_tensorstore(spec)
    elif callable(like):
        return like
    else:
        raise TypeError(f"Can't deserialize {type(like)}")


async def _deserialize_named_array(like, spec, axis_mapping, mesh):
    # the main thing we're worried about is deserialized NamedArrays that are not yet arrays but are ShapedDtypeStructs.
    # These don't (currently) have sharding info, but we can infer it from the axes
    if isinstance(like.array, jax.ShapeDtypeStruct):
        sharding = hax.partitioning.sharding_for_axis(like.axes, axis_mapping, mesh)
        array = await array_ser.async_deserialize(sharding, spec, global_shape=like.array.shape, dtype=like.dtype)
        assert sharding.is_equivalent_to(array.sharding, len(like.array.shape))
        return hax.NamedArray(array, like.axes)
    else:
        array = await _deserialize_one_leaf(like.array, spec, axis_mapping, mesh)
        return hax.NamedArray(array, like.axes)


def tree_deserialize_leaves_tensorstore(
    checkpoint_dir, pytree, axis_mapping: Optional[ResourceMapping] = None, mesh: Optional[Mesh] = None
):
    """
    Deserializes a PyTree of Arrays and NamedArrays from a Tensorstore checkpoint, returning a pytree with the same shape
    as the one provided. This method is capable of deserializing NamedArrays that are the result of an eval_shape call
    (i.e. they are not yet arrays but are ShapedDtypeStructs), provided you pass in the axis_mapping and mesh (or
    they are available by context)

    :param checkpoint_dir: the directory containing the tensorstore checkpoint, can be a local path or a GCS path
    :param pytree: the exemplar pytree
    :param axis_mapping: optional, the axis mapping for the NamedArrays (if they are not yet arrays)
    :param mesh: optional, the mesh for the NamedArrays (if they are not yet arrays)

    :return: a pytree with the same shape as the exemplar pytree, but with the arrays deserialized from the checkpoint
    """
    # TODO: support ShapeDtypeStructs that are not NamedArrays
    leaf_key_paths = jax_utils.leaf_key_paths(pytree, is_leaf=is_named_array)
    specs = htu.tree_map(partial(_tensorstore_spec_for, checkpoint_dir), leaf_key_paths)

    deser_partial = functools.partial(_deserialize_one_leaf, axis_mapping=axis_mapping, mesh=mesh)

    futures = jtu.tree_map(deser_partial, pytree, specs, is_leaf=is_named_array)
    leaves, structure = jtu.tree_flatten(futures, is_leaf=is_named_array)

    async def _do_deserialize():
        values = await asyncio.gather(*leaves)
        return jtu.tree_unflatten(structure, values)

    return asyncio.run(_do_deserialize())
