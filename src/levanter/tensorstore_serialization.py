# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import asyncio
import logging
from functools import partial

import jax
import jax.experimental.array_serialization.serialization as array_ser
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tensorstore
from tensorstore import TensorStore

from levanter.utils import jax_utils


logger = logging.getLogger(__name__)


def tree_serialize_leaves_tensorstore(checkpoint_dir, pytree):
    leaf_key_paths = jax_utils.leaf_key_paths(pytree)
    specs = jtu.tree_map(partial(_tensorstore_spec_for, checkpoint_dir), leaf_key_paths)

    # TODO: jax array_ser has a fancy async manager thing to checkpoint while training, would be good but not right now.
    # array_ser only supports saving sharded arrays, so we can't use its top-level function run_serialization.
    # however we're inspired by its implementation, meaning we'll make a tree of futures and wait on them.
    async def _do_serialize():
        futures = jtu.tree_map(_serialize_one_leaf, pytree, specs)
        return await asyncio.gather(*jtu.tree_leaves(futures))

    asyncio.run(_do_serialize())


def _tensorstore_spec_for(checkpoint_dir, key_path: str):
    checkpoint_path = f"{checkpoint_dir}/{key_path.replace('.', '/')}"
    ts_spec = array_ser.get_tensorstore_spec(checkpoint_path)
    return ts_spec


async def _serialize_one_leaf(x, spec):
    if isinstance(x, jax.Array):
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
    # we have.
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


async def _deserialize_one_leaf(like, spec):
    if isinstance(like, jax.Array):
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


def tree_deserialize_leaves_tensorstore(checkpoint_dir, pytree):
    leaf_key_paths = jax_utils.leaf_key_paths(pytree)
    specs = jtu.tree_map(partial(_tensorstore_spec_for, checkpoint_dir), leaf_key_paths)

    async def _do_deserialize():
        futures = jtu.tree_map(_deserialize_one_leaf, pytree, specs)
        leaves, structure = jtu.tree_flatten(futures)
        values = await asyncio.gather(*leaves)
        return jtu.tree_unflatten(structure, values)

    return asyncio.run(_do_deserialize())
