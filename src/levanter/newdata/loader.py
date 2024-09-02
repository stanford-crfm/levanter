import logging
import time
from typing import Iterable, Iterator, Optional, TypeVar

import jax
from jax import Array
from jax.sharding import Mesh, PartitionSpec

import haliax as hax
from haliax import is_named_array
from haliax._src.util import index_where
from haliax.partitioning import ResourceMapping

from levanter.data.loader import _stack_tree
from levanter.data.utils import batched
from levanter.newdata.dataset import AsyncDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.background_iterable import BackgroundIterable
from levanter.utils.thread_utils import blocking_wait


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

        Args:
            Batch (hax.Axis): The batch axis
            data (AsyncDataset[Ex]): The dataset to load from
            max_buffered_batches (Optional[int]): The maximum number of batches to buffer. If None, the buffer is unbounded.
             If <0, the buffer is disabled and single threaded operation is used.
            axis_resources (Optional[ResourceMapping]): axis mapping
            prefetch_size (int): The number of batches to prefetch
            mesh (Mesh): The mesh to use

        """
        self.max_buffered_batches = max_buffered_batches
        self.prefetch_size = prefetch_size
        self.axis_resources = axis_resources
        self.data_store = data
        self.mesh = mesh
        self.Batch = Batch

        def _exemplar_shape():
            return blocking_wait(self.data_store.async_getitem(0))

        self._ex_leaves, self._ex_structure = jax.tree_flatten(_exemplar_shape(), is_leaf=is_named_array)

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
        return DataLoaderIterator(self, start_from_batch=start_from_batch)


class DataLoaderIterator(Iterator[Ex]):
    def __init__(self, data_loader: DataLoader, start_from_batch: Optional[int] = None):
        self.dl = data_loader
        self._start_from_batch = start_from_batch
        self.mapping = self.dl.axis_resources
        if self.mapping is None:
            self.mapping = hax.partitioning.current_thread_local_mapping()

        # TODO: bring back non-prefetching version
        buffered_batches = self.dl.max_buffered_batches
        self._batches = iter(BackgroundIterable(self._produce_batches, max_capacity=buffered_batches))

    def __next__(self):
        time_start = time.time()
        out = next(self._batches)
        time_end = time.time()
        if (time_end - time_start) > 0.5:
            logger.info(f"Prefetch wasn't fast enough: {time_end - time_start:.3f}")
        return out

    async def _produce_batches(self):
        batch_number = self._start_from_batch or 0
        total_ex_loaded = 0
        done = False
        while not done:
            next_batch_numbers = []
            for i in range(self.dl.prefetch_size):
                if self.dl.data_store.is_finite():
                    next_end = (batch_number + 1) * self.dl.batch_size
                    available_len = await self.dl.data_store.wait_until_len_at_least(next_end)
                    if available_len < next_end:
                        done = True
                        break

                next_batch_numbers.append(batch_number)
                batch_number += 1

            async for batch in self._retrieve_batches(next_batch_numbers):
                yield batch

            total_ex_loaded += self.dl.batch_size * len(next_batch_numbers)

    async def _retrieve_batches(self, batch_numbers: list[int]):
        with hax.axis_mapping(self.mapping), self.dl.mesh:
            indices_for_this_batch_of_batches: list[int] = []
            for bn in batch_numbers:
                indices_this_batch = range(bn * self.dl.batch_size, (bn + 1) * self.dl.batch_size, 1)
                indices_this_batch_this_process = [indices_this_batch[i] for i in self.dl._local_indices]
                indices_for_this_batch_of_batches.extend(indices_this_batch_this_process)

            time_start = time.time()
            individual_datums = await self.dl.data_store.get_batch(indices_for_this_batch_of_batches)
            time_end = time.time()
            logger.debug(f"Time to get {len(batch_numbers)} batches: {time_end - time_start:.3f}")
            time_start = time.time()
            # reshape to be per batch
            individual_datums = list(batched(individual_datums, len(self.dl._local_indices)))

            # below we're gonna get the indices relative to this batch (i.e. 0 to batch_size)
            index_to_datum = [
                {index: datum for index, datum in zip(self.dl._local_indices, individual_data_batch)}
                for individual_data_batch in individual_datums
            ]

            def get_local_batch(bn: int, begin: int, end: int) -> list:
                # TODO: if we ever do "big data" (i.e. huge examples) we might want to be able to load part of an example
                # which will require support from the datastore (i.e. tensorstore)
                device_batch = _stack_tree(self.dl.Batch.name, [index_to_datum[bn][i] for i in range(begin, end)])
                batch_leaves = hax.tree_util.tree_leaves(device_batch)
                return batch_leaves

            def get_local_data_for_leaf(bn, indices: _TensorSliceIndex, leaf_index: int) -> Array:
                batch_slice = indices[0]
                begin, end, stride = batch_slice.indices(self.dl.batch_size)
                if stride != 1:
                    raise ValueError("Stride must be 1")

                leaf_data = (get_local_batch(bn, begin, end))[leaf_index]

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
                        # TODO: this doesn't work with named axes
                        return leaf_data[(..., *other_indices)]

            for batch_offset, bn in enumerate(batch_numbers):

                def make_global_array_for_leaf(leaf_index, item_leaf_shape: ShapeSpec | NamedShapeSpec):
                    def get_data(indices):
                        return get_local_data_for_leaf(batch_offset, indices, leaf_index)

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
                yield gda_tree

    def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.dl.Batch, self.dl.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.dl.axis_resources)  # type: ignore


def _abstractify(x):
    def _abstractify_array(x):
        if isinstance(x, jax.numpy.ndarray):
            return ShapeSpec(x.shape, x.dtype)
        elif isinstance(x, hax.NamedArray):
            return NamedShapeSpec(x.axes, x.dtype)

        return x

    return hax.tree_util.tree_map(_abstractify_array, x)


def _batchified_shape(Batch, leaf: hax.NamedArray | Array) -> ShapeSpec | NamedShapeSpec:
    if is_named_array(leaf):
        return NamedShapeSpec((Batch,) + leaf.axes, leaf.dtype)
    else:
        return ShapeSpec((Batch.size,) + leaf.shape, leaf.dtype)


def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec) -> PartitionSpec:
    if isinstance(shape_spec, ShapeSpec):  # type: ignore
        batch_name = hax.partitioning.physical_axis_name(self.Batch, self.axis_resources)
        return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
    else:
        return hax.partitioning.pspec_for_axis(shape_spec.shape, self.axis_resources)  # type: ignore
