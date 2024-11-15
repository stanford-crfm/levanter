import functools
import logging
import time
from collections import defaultdict
from typing import AsyncIterator, Callable, Iterable, Iterator, Optional, Tuple, TypeVar

import jax
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree

import haliax as hax
from haliax import is_named_array
from haliax._src.util import index_where
from haliax.partitioning import ResourceMapping

from levanter.data.dataset import AsyncDataset
from levanter.data.utils import batched
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.thread_utils import AsyncIteratorWrapper, blocking_wait


Ex = TypeVar("Ex")

_TensorSliceIndex = tuple[slice, ...]
logger = logging.getLogger(__name__)


class DataLoader(Iterable[Ex]):
    def __init__(
        self,
        Batch: hax.Axis,
        data: AsyncDataset[Ex],
        max_buffered_batches: Optional[int],
        mesh: Mesh,
        axis_resources: Optional[ResourceMapping],
        # this is set heuristically for the typical tokenseqdataset we use. Should probably tune
        prefetch_size: int = 32,
    ):
        """

        TODO: document this

        Args:
            Batch (hax.Axis): The batch axis
            data (AsyncDataset[Ex]): The dataset to load from
            max_buffered_batches (Optional[int]): The maximum number of batches to buffer. If None, the buffer is unbounded.
             If <0, the buffer is disabled and single threaded operation is used.
            axis_resources (Optional[ResourceMapping]): axis mapping
            prefetch_size (int): The number of batches to prefetch at once
            mesh (Mesh): The mesh to use

        """
        self.max_buffered_batches = max_buffered_batches
        self.prefetch_size = prefetch_size
        self.axis_resources = axis_resources
        self.data_store = data
        self.mesh = mesh
        self.Batch = Batch

        with local_cpu_mesh():
            # It's important that all data loading happens CPU side. We might relax this one day.
            self._ex_leaves, self._ex_structure = jax.tree_flatten(
                blocking_wait(self.data_store.getitem_async(0)), is_leaf=is_named_array
            )

        local_device_indices, local_indices = self._compute_local_device_indices()

        self._local_device_indices: dict[jax.Device, range] = local_device_indices
        # this is just the flat indices
        self._local_indices: list[int] = local_indices

    def _compute_local_device_indices(self):
        sharding: jax.sharding.Sharding = hax.partitioning.sharding_for_axis(
            self.Batch.name, self.axis_resources, self.mesh
        )
        # this is a map from devices to the slice of the array that they contain (in the global array)
        local_indices_map = sharding.addressable_devices_indices_map((self.batch_size,))
        # we just want all the indices
        local_device_indices: dict[jax.Device, range] = {
            device1: range(*idx[0].indices(self.batch_size))
            for device1, idx in local_indices_map.items()
            if idx is not None
        }
        local_indices: list[int] = []
        for device, indices in local_device_indices.items():
            local_indices.extend(indices)
        return local_device_indices, local_indices

    @property
    def batch_size(self):
        return self.Batch.size

    def __iter__(self):
        return self.iter_from_step(None)

    def iter_from_step(self, start_from_batch: Optional[int] = None):
        # sometimes we pass in an array for the start_from_batch, so we need to check for that
        start_from_batch = int(start_from_batch) if start_from_batch is not None else None
        return DataLoaderIterator(self, start_from_batch=start_from_batch)


class DataLoaderIterator(Iterator[Ex]):
    def __init__(self, data_loader: DataLoader, start_from_batch: Optional[int] = None):
        self.dl = data_loader
        self._start_from_batch = start_from_batch
        self.mapping = self.dl.axis_resources
        if self.mapping is None:
            self.mapping = hax.partitioning.current_thread_local_mapping()

        buffered_batches = self.dl.max_buffered_batches
        self._batches: Iterator[Ex]
        if buffered_batches == 0:
            self._batches = AsyncIteratorWrapper(self._produce_batches())
        else:
            self._batches = _JaxCpuBackgroundIterator(self._produce_batches, max_capacity=buffered_batches)

    def __next__(self):
        time_start = time.time()
        individual_data_batch = next(self._batches)
        data_for_this_batch = {index: datum for index, datum in zip(self.dl._local_indices, individual_data_batch)}
        batch = self._batchify_local_data(data_for_this_batch)

        time_end = time.time()
        if (time_end - time_start) > 0.5:
            logger.info(f"Prefetch wasn't fast enough: {time_end - time_start:.3f}")
        return batch

    async def _produce_batches(self):
        batch_number = self._start_from_batch or 0
        done = False
        while not done:
            target_next_batch_number = batch_number + self.dl.prefetch_size
            max_achievable_batch_number = await self._dataset_get_available_batch_number(target_next_batch_number)
            if max_achievable_batch_number < target_next_batch_number:
                done = True

            next_batch_numbers = list(range(batch_number, min(target_next_batch_number, max_achievable_batch_number)))

            if len(next_batch_numbers) == 0:
                break

            batch_number = next_batch_numbers[-1] + 1

            async for batch in self._retrieve_batches(next_batch_numbers):
                yield batch

    async def _dataset_get_available_batch_number(self, target_max_batch_number: int) -> int:
        if self.dl.data_store.is_finite():
            next_end = (target_max_batch_number + 1) * self.dl.batch_size
            available_len = await self.dl.data_store.wait_until_len_at_least(next_end)
            max_achievable_batch_number = available_len // self.dl.batch_size

            return max_achievable_batch_number

        return target_max_batch_number

    async def _retrieve_batches(self, batch_numbers: list[int]):
        with local_cpu_mesh():
            time_start = time.time()
            individual_datums_for_each_batch = await self._do_retrieve_batch_of_batches(batch_numbers)
            # reshape to be per batch
            time_end = time.time()
            logger.debug(f"Time to get {len(batch_numbers)} batches: {time_end - time_start:.3f}")

            for data in individual_datums_for_each_batch:
                yield data

    def _batchify_local_data(self, data_for_this_batch: dict[int, Array]):
        cache: dict[tuple[int, int], list[Array | hax.NamedArray]] = {}

        def get_local_batch(begin: int, end: int) -> list:
            if (begin, end) in cache:
                return cache[(begin, end)]

            # TODO: if we ever do "big data" (i.e. huge examples) we might want to be able to load part of an example
            # which will require support from the datastore (i.e. tensorstore)
            device_batch = _stack_tree(self.dl.Batch.name, [data_for_this_batch[i] for i in range(begin, end)])
            batch_leaves = hax.tree_util.tree_leaves(device_batch)

            cache[(begin, end)] = batch_leaves

            return batch_leaves

        def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array:
            batch_slice = indices[0]
            begin, end, stride = batch_slice.indices(self.dl.batch_size)
            if stride != 1:
                raise ValueError("Stride must be 1")

            leaf_data = get_local_batch(begin, end)[leaf_index]

            if isinstance(leaf_data, hax.NamedArray):
                # select out the batch axis
                batch_index = index_where(lambda ax: ax.name == self.dl.Batch.name, leaf_data.axes)
                new_indices = list(indices)
                new_indices[batch_index] = slice(None)
                return leaf_data.array[tuple(new_indices)]
            else:
                other_indices = indices[1:]
                if all(idx == slice(None) for idx in other_indices):
                    return leaf_data
                else:
                    return leaf_data[(..., *other_indices)]

        def make_global_array_for_leaf(leaf_index, item_leaf_shape: ShapeSpec | NamedShapeSpec):
            def get_data(indices):
                return get_local_data_for_leaf(indices, leaf_index)

            raw_array = jax.make_array_from_callback(
                to_raw_shape(item_leaf_shape),
                jax.sharding.NamedSharding(self.dl.mesh, self._pspec_for(item_leaf_shape)),
                get_data,
            )
            if isinstance(item_leaf_shape, NamedShapeSpec):
                return hax.NamedArray(raw_array, item_leaf_shape.shape)
            else:
                return raw_array

        gda_leaves = [
            make_global_array_for_leaf(leaf_index, _batchified_shape(self.dl.Batch, item_leaf))
            for leaf_index, item_leaf in enumerate(self.dl._ex_leaves)
        ]
        gda_tree = jax.tree.unflatten(self.dl._ex_structure, gda_leaves)
        return gda_tree

    async def _do_retrieve_batch_of_batches(self, batch_numbers):
        indices_for_this_batch_of_batches: list[int] = []
        for bn in batch_numbers:
            indices_this_batch = range(bn * self.dl.batch_size, (bn + 1) * self.dl.batch_size, 1)
            indices_this_batch_this_process = [indices_this_batch[i] for i in self.dl._local_indices]
            indices_for_this_batch_of_batches.extend(indices_this_batch_this_process)
        individual_datums = await self.dl.data_store.get_batch(indices_for_this_batch_of_batches)
        individual_datums_for_each_batch = list(batched(individual_datums, len(self.dl._local_indices)))
        return individual_datums_for_each_batch

    def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.dl.Batch, self.dl.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.dl.axis_resources)  # type: ignore


def _batchified_shape(Batch, leaf: hax.NamedArray | Array) -> ShapeSpec | NamedShapeSpec:
    if is_named_array(leaf):
        return NamedShapeSpec((Batch,) + leaf.axes, leaf.dtype)
    else:
        return ShapeSpec((Batch.size,) + leaf.shape, leaf.dtype)


class _JaxCpuBackgroundIterator(BackgroundIterator[Ex]):
    """
    We want the thread to only use the CPU device.
    """

    def __init__(self, producer_fn: Callable[[], Iterator[Ex] | AsyncIterator[Ex]], max_capacity: Optional[int]):
        super().__init__(producer_fn, max_capacity)

    def _fill_queue_with_batches(self):
        with local_cpu_mesh():
            super()._fill_queue_with_batches()


@functools.partial(jax.jit, static_argnums=(0,))
def _stack_tree(batch_name, individual_datums):
    def _stack_leaves_unchecked(*leaves):
        if is_named_array(leaves[0]):
            return hax.stack(batch_name, leaves)
        else:
            return jnp.stack(leaves)

    return jax.tree.map(_stack_leaves_unchecked, *individual_datums, is_leaf=is_named_array)


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
