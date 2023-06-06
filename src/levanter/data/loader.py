import abc
import itertools
import logging
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.array import ArrayImpl
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, PyTree

import haliax as hax
import levanter.mesh
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array
from levanter.data import Dataset
from levanter.data.dataset import ShardableDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.py_utils import non_caching_cycle


Ex = TypeVar("Ex")

logger = logging.getLogger(__name__)

# TODO: write tests to verify this works when data spans multiple processes

_TensorSliceIndex = Tuple[slice, ...]


class BatchLoader(Iterable[Ex]):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[Ex]:
        ...

    def batch_size(self) -> int:
        return self.Batch.size

    Batch: hax.Axis


class ShardedBatchLoader(BatchLoader[Ex]):
    """
    ShardedBatchLoader wraps a "local dataset" (a dataset that is shardable and can be iterated over) to produce
    distributed/sharded jax.Arrays representing batches of data. Each array that has a global shape
    but only has the data for some of the chunks of the array (namely, the ones on the local devices).
    Thus, each process loads the data for its devices.

    **NOTE: ShardedBatchLoader loops forever since it's hard to do coordination.**

    The details are a bit complex: We have a device mesh of shape (data, model). We want each row of the device mesh to
    get batch_size//num_rows examples. Usually, a process will be responsible for one or more entire rows, meaning
    that it wil load data that is distinct from every other process. However, if num_cols > num_devices_per_process,
    then some processes will need to load the same data. We use the process_mesh_position to determine which data to
    load, by determining which row(s) of the device mesh the process is responsible for.

    :arg local_dataset: a dataset that is shardable and can be iterated over
    :arg mesh: the device mesh
    :arg Batch: the batch size
    """

    def __init__(
        self,
        local_dataset: ShardableDataset[Ex],
        mesh: Mesh,
        Batch: hax.Axis,
        axis_resources: Optional[ResourceMapping] = None,
        *,
        override_process_data_pos: Optional[int] = None,  # for testing
        override_process_data_groups: Optional[int] = None,  # for testing
    ):
        self.mesh = mesh
        self.Batch = Batch
        self.axis_resources = axis_resources

        process_data_pos = override_process_data_pos or levanter.mesh.process_mesh_position(mesh)[0]
        num_data_process_groups = override_process_data_groups or levanter.mesh.process_mesh_size(mesh)[0]

        if not override_process_data_groups:
            assert num_data_process_groups <= jax.process_count()

        self.process_data_pos = process_data_pos
        self.num_data_process_groups = num_data_process_groups
        assert self.Batch.size % num_data_process_groups == 0

        self.item_dataset = local_dataset.shard(process_data_pos, num_data_process_groups)

    def __iter__(self) -> Iterator[PyTree[jax.Array]]:
        one_item_generator = non_caching_cycle(self.item_dataset)

        for i, item in enumerate(one_item_generator):
            # ok this is a bit messy: we want to create a batch of items from our dataset, only loading
            # the relevant data for each process.
            # In general an item is represented as a PyTree, whose leaves are (named or unnamed) arrays.
            # To make a batch we just want to add a leading dimension to each leaf array by stacking.
            # That is, we have (conceptually) a List[PyTree[Array]] and we want to produce a PyTree[List[Array]]
            # The difference is that we want to do this in a way that only loads the relevant data for each process
            # So it's more that we have a LocalBatch[PyTree[Array]] and we want to produce a PyTree[GlobalBatch[Array]]
            # because more than one device can get the same data, we need to make sure we only load it once since we're
            # streaming. This is the cache
            local_batch_leaves: Dict[Tuple[int, int], List[Array]] = {}  # batch indices -> list of items

            batch_offset = self.process_data_pos * self.local_batch_size
            local_batch = list(itertools.islice(one_item_generator, self.local_batch_size))
            batch_tree_structure = None

            def get_local_batch(begin: int, end: int) -> List[Array]:
                nonlocal batch_tree_structure

                key = (begin, end)
                if key in local_batch_leaves:
                    return local_batch_leaves[key]

                individual_datums = local_batch[(begin - batch_offset) : (end - batch_offset)]

                device_batch = jax.tree_map(self._stack_leaves_unchecked, *individual_datums, is_leaf=is_named_array)
                batch_leaves, _batch_structure = jax.tree_util.tree_flatten(device_batch)

                if batch_tree_structure is None:
                    batch_tree_structure = _batch_structure

                local_batch_leaves[key] = batch_leaves

                return batch_leaves

            shape_leaves = jax.tree_util.tree_leaves(self.item_shape)

            # Callback passed to jax.make_array_from_callback to get the data for each device
            def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array:
                batch_slice = indices[0]
                begin, end, _ = batch_slice.indices(self.Batch.size)
                local_batch = get_local_batch(begin, end)
                leaf = local_batch[leaf_index]
                return leaf[(..., *indices[1:])]

            # TODO: with a bit more fanciness, we can avoid needing the item_shape
            gda_leaves = [
                jax.make_array_from_callback(
                    to_raw_shape(shape),
                    jax.sharding.NamedSharding(self.mesh, self._pspec_for(shape)),
                    lambda indices: get_local_data_for_leaf(indices, leaf_index),
                )
                for leaf_index, shape in enumerate(shape_leaves)
            ]

            gda_tree = jax.tree_util.tree_unflatten(batch_tree_structure, gda_leaves)

            if i % 100 == 0 and logger.getEffectiveLevel() <= logging.DEBUG:
                for leaf in gda_leaves:
                    check_sharded_consistency(leaf, True)

            yield gda_tree  # type: ignore

    def _pspec_for(self, shape_spec: Union[ShapeSpec, NamedShapeSpec]) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.Batch, self.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.axis_resources)  # type: ignore

    @staticmethod
    def _get_begin_end_for_slice(tensor_shape, tslice_index) -> Tuple[Tuple[int, int], ...]:
        # begin, end, step
        my_indices: Tuple[Tuple[int, int, int], ...] = tuple(
            s.indices(axis_size) for axis_size, s in zip(tensor_shape, tslice_index)
        )
        assert all(s[2] == 1 for s in my_indices)  # ensure step is 1
        return tuple(s[0:2] for s in my_indices)

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return _batchify_item_shape(self.item_dataset.item_shape, self.Batch)

    @property
    def batch_size(self) -> int:
        """Returns the 'global' batch size: the effective number of examples in a batch across all devices/hosts"""
        return self.Batch.size

    @property
    def local_batch_size(self) -> int:
        """Returns the 'local' batch size: the number of examples in a batch on this host"""
        return self.batch_size // self.num_data_process_groups

    def _stack_leaves_unchecked(self, *leaves):
        assert len(leaves) <= self.Batch.size
        assert self.Batch.size % len(leaves) == 0

        if is_named_array(leaves[0]):
            with hax.enable_shape_checks(False):  # because we're building parts of the array on each device
                return hax.stack(self.Batch, leaves)
        else:
            return np.stack(leaves)


class ReplicatedBatchLoader(BatchLoader[Ex]):
    """A batch loader that creates batches without sharded data loading. All examples are loaded on all machines and then
    sharded. This is useful if you have a small dataset and want to make a single pass over it.

    Note: this class discards the final batch if it is smaller than the batch size.
    """

    def __init__(
        self,
        item_dataset: Dataset[Ex],
        mesh: Mesh,
        Batch: hax.Axis,
        axis_resources: Optional[ResourceMapping] = None,
    ):
        self.local_dataset = item_dataset
        self.mesh = mesh
        self.Batch = Batch
        self.axis_resources = axis_resources

    def __iter__(self):
        item_iter = iter(self.local_dataset)
        for batch in self._batched(item_iter):
            stacked = jax.tree_map(lambda *leaves: self._stack_leaves(*leaves), *batch, is_leaf=is_named_array)
            yield self._shard(stacked)

    def _batched(self, item_iter):
        batch = []
        for item in item_iter:
            batch.append(item)
            if len(batch) == self.Batch.size:
                yield batch

    def _stack_leaves(self, *leaves):
        assert len(leaves) == self.Batch.size

        if is_named_array(leaves[0]):
            return hax.stack(self.Batch, leaves)
        else:
            return np.stack(leaves)

    def _shard(self, batch):
        def _shard_leaf(leaf):
            pspec = self._pspec_for(leaf)
            with self.mesh:
                return pjit(lambda x: x, in_axis_resources=None, out_axis_resources=pspec)(leaf)

        return jax.tree_map(_shard_leaf, batch, is_leaf=is_named_array)

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return _batchify_item_shape(self.local_dataset.item_shape, self.Batch)

    def _pspec_for(self, leaf) -> PartitionSpec:
        if not isinstance(leaf, hax.NamedArray):
            batch_name = hax.partitioning.physical_axis_name(self.Batch, self.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(leaf.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(leaf.axes, self.axis_resources)


def _batchify_item_shape(item_shape: PyTree[Union[ShapeSpec, NamedShapeSpec]], Batch: hax.Axis):
    def _batchify_shape_spec(shape_spec: Union[ShapeSpec, NamedShapeSpec]):
        shape = shape_spec.shape
        assert shape is not None, "item_shape must have a fully determined shape to work with batching"
        if isinstance(shape_spec, NamedShapeSpec):
            return NamedShapeSpec((Batch,) + shape, shape_spec.dtype)
        else:
            return ShapeSpec((Batch.size,) + shape, shape_spec.dtype)

    return jax.tree_map(_batchify_shape_spec, item_shape)


def check_sharded_consistency(tree: PyTree[ArrayImpl], check_disjoint_indices_are_different: bool = False):
    """Checks the following consistency conditions on an array:
    - all replicas have the same data
    - if check_disjoint_indices_are_different is True, then all shards with disjoint indices have different data
    """
    # index is a tuple[slice, ...], slices are obnoxiously not hashable so we have to convert to tuple

    def check_array(array: ArrayImpl):
        def _to_tuple(index: Tuple[slice, ...]) -> Tuple[Tuple[int, int], ...]:
            my_indices: Tuple[Tuple[int, int], ...] = tuple(
                s.indices(axis_size)[0:2] for axis_size, s in zip(array.shape, index)
            )

            return my_indices

        replicas_by_index = defaultdict(list)
        for shard in array.global_shards:
            replicas_by_index[_to_tuple(shard.index)].append(shard)

        # global shards is not necessarily sorted consistently, so we have to sort the indices
        sorted_indices = sorted(replicas_by_index.keys())

        # ok now get canonical versions of each index
        replica_0_arrays = {}

        for index in sorted_indices:
            shards = replicas_by_index[index]
            try:
                leader = next(s for s in shards if s.replica_id == 0)
            except StopIteration:
                raise ValueError("No replica 0 for index", index)

            data = leader.data
            if data is None:
                shard_shape = [s[1] - s[0] for s in index]
                data = jnp.zeros(shard_shape, dtype=array.dtype)

            replica_0_array = multihost_utils.broadcast_one_to_all(data, is_source=leader.data is not None)
            replica_0_arrays[index] = replica_0_array

        for shard in array.addressable_shards:
            replica_0_array = replica_0_arrays[_to_tuple(shard.index)]
            assert shard.data is not None

            if not jnp.array_equal(shard.data, replica_0_array, equal_nan=True):
                raise ValueError("Shard data does not match replica 0 data", shard, replica_0_array)

            if check_disjoint_indices_are_different:
                for other_index, other_array in replica_0_arrays.items():
                    if other_index == _to_tuple(shard.index):
                        continue

                    if shard.index != other_index:
                        if jnp.array_equal(shard.data, other_array, equal_nan=True):
                            raise ValueError(
                                "Shard data is the same as another shard with disjoint indices", shard, other_array
                            )

    for leaf in jax.tree_util.tree_leaves(tree):
        check_array(leaf)
