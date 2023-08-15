import abc
import functools
import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, PyTree

import haliax as hax
from haliax import NamedArray
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array

import levanter.mesh
from levanter.data import Dataset
from levanter.data.dataset import ShardableDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.background_iterable import BackgroundIterable
from levanter.utils.py_utils import non_caching_cycle


Ex = TypeVar("Ex")

logger = logging.getLogger(__name__)

# TODO: write tests to verify this works when data spans multiple processes

_TensorSliceIndex = Tuple[slice, ...]


class BatchLoader(Iterable[Ex], abc.ABC):
    Batch: hax.Axis
    mesh: Mesh
    axis_resources: Optional[ResourceMapping]

    def __init__(self, max_capacity: Optional[int], axis_resources: Optional[ResourceMapping]):
        """
        :param max_capacity: if not None, the maximum number of batches to keep in memory at once. If <0 then load in the main thread
        :param axis_resources:
        """
        self.max_capacity = max_capacity
        self.axis_resources = axis_resources

    def __iter__(self) -> Iterator[Ex]:
        ax_resources = self.axis_resources
        if ax_resources is None:
            ax_resources = hax.partitioning.current_thread_local_mapping()

        def produce_batches():
            with hax.axis_mapping(ax_resources):
                for batch in self._produce_batches():
                    yield batch

        if self.max_capacity is not None and self.max_capacity < 0:
            yield from produce_batches()
        else:
            bg_iter = BackgroundIterable(produce_batches, max_capacity=self.max_capacity)
            yield from bg_iter

    @abc.abstractmethod
    def _produce_batches(self) -> Iterator[Ex]:
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        return self.Batch.size

    def _construct_global_array_for_tree(self, item_exemplar: PyTree, get_batch_items: Callable[[int, int], PyTree]):
        # ok this is a bit messy: we want to create a batch of items from our dataset, only loading
        # the relevant data for each process.
        # In general an item is represented as a PyTree, whose leaves are (named or unnamed) arrays.
        # To make a batch we just want to add a leading dimension to each leaf array by stacking.
        # That is, we have (conceptually) a List[PyTree[Array]] and we want to produce a PyTree[List[Array]]
        # The difference is that we want to do this in a way that only loads the relevant data for each process
        # So it's more that we have a LocalBatch[PyTree[Array]] and we want to produce a PyTree[GlobalBatch[Array]]
        # because more than one device can get the same data, we need to make sure we only load it once since we're
        # streaming. This is the cache
        stacked_local_batch: Dict[Tuple[int, int], List[Array | hax.NamedArray]] = {}

        def get_local_batch(begin: int, end: int) -> List[Array]:
            key = (begin, end)
            if key in stacked_local_batch:
                return stacked_local_batch[key]

            individual_datums = get_batch_items(begin, end)

            device_batch = _stack_tree(self.Batch.name, individual_datums)
            batch_leaves = jtu.tree_leaves(device_batch)

            stacked_local_batch[key] = batch_leaves

            return batch_leaves

        def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array:
            batch_slice = indices[0]
            begin, end, _ = batch_slice.indices(self.Batch.size)
            local_batch = get_local_batch(begin, end)
            leaf = local_batch[leaf_index]
            other_indices = indices[1:]
            if all(idx == slice(None) for idx in other_indices):
                return leaf
            else:
                return leaf[(..., *indices[1:])]

        def make_global_array_for_leaf(leaf_index, item_leaf_shape: Union[ShapeSpec, NamedShapeSpec]):
            raw_array = jax.make_array_from_callback(
                to_raw_shape(item_leaf_shape),
                jax.sharding.NamedSharding(self.mesh, self._pspec_for(item_leaf_shape)),
                lambda indices: get_local_data_for_leaf(indices, leaf_index),
            )
            if isinstance(item_leaf_shape, NamedShapeSpec):
                return NamedArray(raw_array, item_leaf_shape.shape)
            else:
                return raw_array

        item_leaves, item_shape = jtu.tree_flatten(item_exemplar, is_leaf=is_named_array)

        gda_leaves = [
            make_global_array_for_leaf(leaf_index, _batchified_shape(self.Batch, item_leaf))
            for leaf_index, item_leaf in enumerate(item_leaves)
        ]

        gda_tree = jtu.tree_unflatten(item_shape, gda_leaves)

        return gda_tree

    def _pspec_for(self, shape_spec: Union[ShapeSpec, NamedShapeSpec]) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.Batch, self.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.axis_resources)  # type: ignore


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
    :param max_capacity: if not None, the maximum number of batches to keep in memory at once. If <0 then load in the main thread
    """

    def __init__(
        self,
        local_dataset: ShardableDataset[Ex],
        mesh: Mesh,
        Batch: hax.Axis,
        axis_resources: Optional[ResourceMapping] = None,
        max_capacity: Optional[int] = 10,
        *,
        override_process_data_pos: Optional[int] = None,  # for testing
        override_process_data_groups: Optional[int] = None,  # for testing
    ):
        self.mesh = mesh
        self.Batch = Batch

        process_data_pos = override_process_data_pos or levanter.mesh.process_mesh_position(mesh)[0]
        num_data_process_groups = override_process_data_groups or levanter.mesh.process_mesh_size(mesh)[0]

        if not override_process_data_groups:
            assert num_data_process_groups <= jax.process_count()

        self.process_data_pos = process_data_pos
        self.num_data_process_groups = num_data_process_groups
        assert self.Batch.size % num_data_process_groups == 0

        self.item_dataset = local_dataset.shard(process_data_pos, num_data_process_groups)
        super().__init__(max_capacity, axis_resources)

    def _produce_batches(self) -> Iterator[PyTree]:
        one_item_generator = non_caching_cycle(self.item_dataset)
        batched = _batched(one_item_generator, self.local_batch_size)

        while True:
            batch_offset = self.process_data_pos * self.local_batch_size
            local_batch: List[PyTree] = next(batched)

            batch = self._construct_global_array_for_tree(
                item_exemplar=local_batch[0],
                get_batch_items=lambda begin, end: local_batch[(begin - batch_offset) : (end - batch_offset)],
            )

            yield batch

    @property
    def batch_size(self) -> int:
        """Returns the 'global' batch size: the effective number of examples in a batch across all devices/hosts"""
        return self.Batch.size

    @property
    def local_batch_size(self) -> int:
        """Returns the 'local' batch size: the number of examples in a batch on this host"""
        return self.batch_size // self.num_data_process_groups


@functools.partial(jax.jit, static_argnums=(0,))
def _stack_tree(batch_name, individual_datums):
    def _stack_leaves_unchecked(*leaves):
        if is_named_array(leaves[0]):
            return hax.stack(batch_name, leaves)
        else:
            return jnp.stack(leaves)

    return jax.tree_map(_stack_leaves_unchecked, *individual_datums, is_leaf=is_named_array)


class ReplicatedBatchLoader(BatchLoader[Ex]):
    """A batch loader that creates batches without sharded data loading. All examples are loaded on all machines and then
    sharded. This is useful if you have a small dataset and want to make a single pass over it.

    Note: this class discards the final batch if it is smaller than the batch size.

    :arg item_dataset: a dataset that is shardable and can be iterated over
    :arg mesh: the device mesh
    :arg Batch: the batch size
    :arg axis_resources: the resources for the batch axis
    :param max_capacity: if not None, the maximum number of batches to keep in memory at once. If <0 then load in the main thread
    """

    def __init__(
        self,
        item_dataset: Dataset[Ex],
        mesh: Mesh,
        Batch: hax.Axis,
        axis_resources: Optional[ResourceMapping] = None,
        max_capacity: Optional[int] = 10,
    ):
        self.item_dataset = item_dataset
        self.mesh = mesh
        self.Batch = Batch

        super().__init__(max_capacity, axis_resources)

    def _produce_batches(self):
        for batch in _batched(self.item_dataset, self.Batch.size):
            sharded = self._construct_global_array_for_tree(
                item_exemplar=batch[0], get_batch_items=lambda begin, end: batch[begin:end]
            )
            yield sharded


def _batchified_shape(Batch, leaf: Union[NamedArray, Array]):
    if isinstance(leaf, NamedArray):
        return NamedShapeSpec((Batch,) + leaf.axes, leaf.dtype)
    else:
        return ShapeSpec((Batch.size,) + leaf.shape, leaf.dtype)


def check_sharded_consistency(tree: PyTree, check_disjoint_indices_are_different: bool = False):
    """Checks the following consistency conditions on an array:
    - all replicas have the same data
    - if check_disjoint_indices_are_different is True, then all shards with disjoint indices have different data
    """
    # index is a tuple[slice, ...], slices are obnoxiously not hashable so we have to convert to tuple

    def check_array(array):
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

    for leaf in jtu.tree_leaves(tree):
        check_array(leaf)


def _batched(item_iter, size):
    batch = []
    for item in item_iter:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
