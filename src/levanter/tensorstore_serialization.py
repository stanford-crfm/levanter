# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import asyncio
import functools
import logging
import os
from functools import partial
from typing import Optional

import jax
import jax.experimental.array_serialization.serialization as array_ser
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tensorstore
from jax.sharding import Mesh
from tensorstore import TensorStore

import haliax as hax
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array

from levanter.utils import jax_utils


logger = logging.getLogger(__name__)


def tree_serialize_leaves_tensorstore(checkpoint_dir, pytree):
    leaf_key_paths = jax_utils.leaf_key_paths(pytree, is_leaf=is_named_array)
    specs = jtu.tree_map(partial(_tensorstore_spec_for, checkpoint_dir), leaf_key_paths, is_leaf=is_named_array)

    # TODO: jax array_ser has a fancy async manager thing to checkpoint while training, would be good but not right now.
    # array_ser only supports saving sharded arrays, so we can't use its top-level function run_serialization.
    # however we're inspired by its implementation, meaning we'll make a tree of futures and wait on them.
    async def _do_serialize():
        futures = jtu.tree_map(_serialize_one_leaf, pytree, specs, is_leaf=is_named_array)
        return await asyncio.gather(*jtu.tree_leaves(futures))

    asyncio.run(_do_serialize())


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
    return await t.read("C")


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
    specs = jtu.tree_map(partial(_tensorstore_spec_for, checkpoint_dir), leaf_key_paths, is_leaf=is_named_array)

    deser_partial = functools.partial(_deserialize_one_leaf, axis_mapping=axis_mapping, mesh=mesh)

    async def _do_deserialize():
        futures = jtu.tree_map(deser_partial, pytree, specs, is_leaf=is_named_array)
        leaves, structure = jtu.tree_flatten(futures, is_leaf=is_named_array)
        values = await asyncio.gather(*leaves)
        return jtu.tree_unflatten(structure, values)

    return asyncio.run(_do_deserialize())
