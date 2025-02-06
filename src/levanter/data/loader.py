import dataclasses
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import AsyncIterator, Callable, Generic, Iterable, Iterator, Optional, Tuple, TypeVar

import equinox
import jax
from jax import Array
from jax import numpy as jnp
from jax import tree_util as jtu
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree
from optax.tree_utils import tree_zeros_like

import haliax as hax
from haliax import is_named_array
from haliax._src.util import index_where
from haliax.partitioning import ResourceMapping

from levanter.data.dataset import AsyncDataset
from levanter.data.utils import batched
from levanter.models.attention import AttentionMask
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule, IntSchedule
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.background_iterable import BackgroundIterator
from levanter.utils.jax_utils import local_cpu_mesh, use_cpu_device
from levanter.utils.thread_utils import AsyncIteratorWrapper, blocking_wait


Ex = TypeVar("Ex")

_TensorSliceIndex = tuple[slice, ...]
logger = logging.getLogger(__name__)


def local_indices_for_bs_and_sharding(sharding: jax.sharding.Sharding, batch_size: int) -> dict[jax.Device, range]:
    """Return the local indices for each device for a given batch size and sharding."""
    local_indices_map = sharding.addressable_devices_indices_map((batch_size,))
    # we just want all the indices
    local_device_indices: dict[jax.Device, range] = {
        device: range(*idx[0].indices(batch_size)) for device, idx in local_indices_map.items() if idx is not None
    }
    return local_device_indices


# NOTE: In general there are a lot of different indices flying around. Here's a quick guide:
# - `step` or `batch_number` or `bn` is the training step or batch number
# - `global` indices refer to the index into the datastore
# - `local` indices refer to the index into the batch. These start at 0 for every batch and go up to the batch size.

# That is, for each batch, we have:
# global_offset = global_offset_at_step(bn)
# batch_size = batch_size_at_step(bn)
# local_data_to_global_map = {i: global_offset + i for i in range(batch_size)}

# But note that we only load the data we need for our local devices, so we only load a subset of the data for each batch.


@dataclass
class _Batch(Generic[Ex]):
    index: int
    global_data_offset: int
    global_size: int
    data_by_local_index: dict[int, Ex]


class DataLoader(Iterable[Ex]):
    def __init__(
        self,
        data: AsyncDataset[Ex],
        batch_size: int | IntSchedule | hax.Axis,
        *,
        batch_axis_name: str | None = None,
        max_buffered_batches: Optional[int] = 64,
        mesh: Mesh,
        axis_resources: Optional[ResourceMapping],
        prefetch_size: int = 32,
    ):
        """
        Batch- and NamedArray-aware data loader. This class works with an [AsyncDataset][], a Mesh,
        and a resource mapping to load data in a way that is aware of the batch axis and the sharding
        of the data. In general, each machine only loads the data that it needs to process.

        (At the moment, it's not smart enough to load parts of examples if examples are sharded across machines,
        but whole examples are handled correctly.)

        Args:
            batch_size (int | IntSchedule | None): The size of the batch or a schedule for the size of the batch
            data (AsyncDataset[Ex]): The dataset to load from
            max_buffered_batches (Optional[int]): The maximum number of batches to buffer. If None, the buffer is unbounded.
             If <0, the buffer is disabled and single threaded operation is used.
            axis_resources (Optional[ResourceMapping]): axis mapping
            prefetch_size (int): The number of batches to prefetch at once
            mesh (Mesh): The mesh to use
            batch_axis_name (str | None): The name of the batch axis. If None, defaults to "batch" unless batch_size is an Axis.

        """
        self.max_buffered_batches = max_buffered_batches
        self.prefetch_size = prefetch_size
        self.axis_resources = axis_resources
        self.data_store = data
        self.mesh = mesh

        if isinstance(batch_size, hax.Axis):
            assert batch_axis_name is None
            self.batch_axis_name = batch_size.name
            self.scheduler = BatchSchedule(batch_size.size)
        else:
            self.batch_axis_name = batch_axis_name or "batch"
            self.scheduler = BatchSchedule(batch_size)

        self._batch_sharding = hax.partitioning.sharding_for_axis(self.batch_axis_name, axis_resources, mesh)

        with local_cpu_mesh():
            # It's important that all data loading happens CPU side. We might relax this one day.
            self._ex_leaves, self._ex_structure = jax.tree_flatten(
                blocking_wait(self.data_store.getitem_async(0)), is_leaf=is_named_array
            )

    def local_data_indices_by_device_for_step(self, step: int) -> dict[jax.Device, range]:
        """

        Returns the 0-based indices of the data that should be loaded by each device for a given step.

        That is, if we are on step i with a fixed batch size of b, then device d should load the data at indices
        [step * bs + (d * local_bs), step * bs + ((d + 1) * local_bs) ) where local_bs is the number of data points
        that device d should load.
        """
        batch_size = self.scheduler.batch_size_at_step(step)
        local_indices = self.compute_local_device_indices_for_bs(batch_size)

        return local_indices

    def global_data_indices_by_device_for_step(self, step: int) -> dict[jax.Device, range]:
        local_indices = self.local_data_indices_by_device_for_step(step)
        offset = self.scheduler.global_data_offset_by_step(step)

        return {device: range(offset + r.start, offset + r.stop, r.step) for device, r in local_indices.items()}

    def batch_axis_at_step(self, step: int):
        size = self.scheduler.batch_size_at_step(step)

        return hax.Axis(self.batch_axis_name, size)

    @lru_cache
    def compute_local_device_indices_for_bs(self, bs) -> dict[jax.Device, range]:
        return local_indices_for_bs_and_sharding(self._batch_sharding, bs)

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
        batch = self._batchify_local_data(individual_data_batch)

        time_end = time.time()
        if (time_end - time_start) > 0.5:
            logger.info(f"Prefetch wasn't fast enough: {time_end - time_start:.3f}")
        return batch

    async def _produce_batches(self):
        batch_number = self._start_from_batch or 0
        done = False
        while not done:
            # we try to prefetch multiple batches at a time
            target_next_batch_number = batch_number + self.dl.prefetch_size
            max_achievable_batch_number = await self._dataset_get_available_batch_number(target_next_batch_number)
            if max_achievable_batch_number < target_next_batch_number:
                done = True

            next_batch_numbers = list(range(batch_number, min(target_next_batch_number, max_achievable_batch_number)))

            if len(next_batch_numbers) == 0:
                break

            time_start = time.time()
            batch_of_batches: list[_Batch[Ex]] = await self._do_retrieve_batch_of_batches(next_batch_numbers)
            time_end = time.time()
            logger.debug(f"Time to get {len(next_batch_numbers)} batches: {time_end - time_start:.3f}")

            for batch in batch_of_batches:
                yield batch

            batch_number = next_batch_numbers[-1] + 1

    async def _dataset_get_available_batch_number(self, target_max_batch_number: int) -> int:
        """
        Wait until the data store has enough data to support the given batch number. If
        the data store is finite, this will wait until the data store has at least `target_max_batch_number` batches
        or until the data store is exhausted, in which case it will return the last batch number that the data store
        has data for.
        """
        if self.dl.data_store.is_finite():
            next_end = self.dl.scheduler.global_data_offset_by_step(target_max_batch_number)
            available_len = await self.dl.data_store.wait_until_len_at_least(next_end)

            while available_len < next_end:
                # backoff until we find a batch size we can support
                # TODO: we could be much smarter about this but unlikely to be a bottle neck
                target_max_batch_number -= 1
                next_end = self.dl.scheduler.global_data_offset_by_step(target_max_batch_number)
                if target_max_batch_number < 0:
                    raise ValueError("No data available")

        return target_max_batch_number

    def _batchify_local_data(self, batch: _Batch[Ex]) -> Ex:
        """
        Stacks the individual examples (pytrees) into a single example (pytree) with the batch axis added
        and creates a global array for each leaf of the example.
        """
        cache: dict[tuple[int, int], list[Array | hax.NamedArray]] = {}
        Batch = hax.Axis(self.dl.batch_axis_name, batch.global_size)

        def get_local_batch(begin: int, end: int) -> list:
            if (begin, end) in cache:
                return cache[(begin, end)]

            # TODO: if we ever do "big data" (i.e. huge examples) we might want to be able to load part of an example
            # which will require support from the datastore (i.e. tensorstore)
            device_batch = stack_tree(
                self.dl.batch_axis_name, [batch.data_by_local_index[i] for i in range(begin, end)]
            )
            batch_leaves = hax.tree_util.tree_leaves(device_batch)

            cache[(begin, end)] = batch_leaves

            return batch_leaves

        def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array:
            batch_slice = indices[0]
            begin, end, stride = batch_slice.indices(batch.global_size)
            if stride != 1:
                raise ValueError("Stride must be 1")

            leaf_data = get_local_batch(begin, end)[leaf_index]

            if isinstance(leaf_data, hax.NamedArray):
                # select out the batch axis
                batch_index = index_where(lambda ax: ax.name == Batch.name, leaf_data.axes)
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
            make_global_array_for_leaf(leaf_index, _batchified_shape(Batch, item_leaf))
            for leaf_index, item_leaf in enumerate(self.dl._ex_leaves)
        ]
        gda_tree = jax.tree.unflatten(self.dl._ex_structure, gda_leaves)
        return gda_tree

    async def _do_retrieve_batch_of_batches(self, batch_numbers: list[int]) -> list[_Batch[Ex]]:
        """
        Retrieve the data for a batch of batches.

        - For each batch number, we get the mapping from our local devices to the indices in the data store that we need
          to load.
        - We then load the data for all the devices for all the batch numbers.
        - we then return each batch as a dictionary from device to the data for that device.
        """
        global_indices_for_each_batch: list[list[int]] = []
        sizes = [self.dl.scheduler.batch_size_at_step(bn) for bn in batch_numbers]
        offsets = []

        for bn in batch_numbers:
            global_offset = self.dl.scheduler.global_data_offset_by_step(bn)
            offsets.append(global_offset)
            local_indices_for_device = self.dl.local_data_indices_by_device_for_step(bn)

            distinct_local_indices_this_batch = set()
            for indices in local_indices_for_device.values():
                for local_index in indices:
                    distinct_local_indices_this_batch.add(local_index)

            global_indices_for_this_batch = [global_offset + i for i in distinct_local_indices_this_batch]
            global_indices_for_each_batch.append(global_indices_for_this_batch)

        # flattened view so we can load all the data at once
        indices_for_this_batch_of_batches: list[int] = [
            i for indices in global_indices_for_each_batch for i in indices
        ]
        individual_datums = await self.dl.data_store.get_batch(indices_for_this_batch_of_batches)

        # unflatten
        global_map: dict[int, Ex] = {}
        for i, datum in enumerate(individual_datums):
            global_map[indices_for_this_batch_of_batches[i]] = datum

        out: list[_Batch[Ex]] = []

        for bn, sz, offset, global_indices_batch in zip(batch_numbers, sizes, offsets, global_indices_for_each_batch):
            local_index_to_example = {}
            for global_index in global_indices_batch:
                local_index = global_index - offset
                local_index_to_example[local_index] = global_map[global_index]

            out.append(_Batch(bn, offset, sz, local_index_to_example))

        return out

    def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.dl.batch_axis_name, self.dl.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.dl.axis_resources)  # type: ignore


def _make_dummy_instance(batch, Pos):
    """
    Creates a dummy instance matching the shape of the provided batch.
    If the dataset is exhausted and a full batch is needed, this function returns a dummy instance
    with all elements set to zero and a segment mask filled with -1. This design ensures that the dummy
    instance does not contribute to the loss during training.
    """
    dummy_instance: LmExample = tree_zeros_like(batch[0])
    dummy_segment_mask = hax.full(Pos, -1, dtype=jnp.int32)
    dummy_attn = AttentionMask.causal().with_segment_ids(dummy_segment_mask)
    dummy_instance = dataclasses.replace(dummy_instance, attn_mask=dummy_attn)
    return dummy_instance


def stack_batches(example_iterator, Pos, Batch):
    """
    Stack examples from an iterator into a batch.

    Args:
        Batch: The batch axis.
        Pos: The position axis.
        example_iterator: An iterator of examples.

    Returns:
        A batch of examples.
    """
    # add timer here as well and profile
    with use_cpu_device():
        for batch in batched(example_iterator, Batch.size):
            if len(batch) < Batch.size:
                dummy_instance = _make_dummy_instance(batch, Pos)
                batch.extend([dummy_instance] * (Batch.size - len(batch)))
            yield stack_tree(Batch, batch)


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


@equinox.filter_jit
def stack_tree(batch_name, individual_datums):
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
