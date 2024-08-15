import asyncio
import logging
import threading
from typing import Iterable, Iterator, Optional, TypeVar

import jax
from jax import Array
from jax.sharding import Mesh, PartitionSpec

import haliax as hax
from haliax import is_named_array
from haliax._src.util import index_where
from haliax.partitioning import ResourceMapping

from levanter.data.loader import _stack_tree
from levanter.newdata.dataset import AsyncDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec, to_raw_shape
from levanter.utils.background_iterable import BackgroundIterable


Ex = TypeVar("Ex")

_TensorSliceIndex = tuple[slice, ...]
logger = logging.getLogger(__name__)


class DataLoader(Iterable[Ex]):
    def __init__(
        self,
        Batch: hax.Axis,
        data: AsyncDataset[Ex],
        max_buffered_items: Optional[int],
        mesh: Mesh,
        axis_resources: Optional[ResourceMapping],
    ):
        """

        Args:
            max_buffered_items (Optional[int]): The maximum number of items to buffer. If None, the buffer is unbounded.
             If <0, the buffer is disabled and single threaded operation is used.
            axis_resources (Optional[ResourceMapping]): axis mapping
        """
        self.max_capacity = max_buffered_items
        self.axis_resources = axis_resources
        self.data_store = data
        self.mesh = mesh
        self.Batch = Batch

        def _exemplar_shape():
            return blocking_wait(self.data_store.async_getitem(0))

        self._ex_leaves, self._ex_structure = jax.tree_flatten(_exemplar_shape(), is_leaf=is_named_array)

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

        if self.dl.max_capacity is not None and self.dl.max_capacity >= 0:
            self._batches = iter(BackgroundIterable(self._produce_batches, max_capacity=self.dl.max_capacity))
        else:
            self._batches = self._produce_batches()

    def __next__(self):
        return next(self._batches)

    def _produce_batches(self):
        batch_number = self._start_from_batch or 0
        total_ex_loaded = 0
        while True:
            if self.dl.data_store.is_finite():
                next_end = (batch_number + 1) * self.dl.batch_size
                available_len = blocking_wait(self.dl.data_store.wait_until_len_at_least(next_end))
                if available_len < next_end:
                    break

            batch = blocking_wait(self._produce_batch(batch_number))
            batch_number += 1
            yield batch

            total_ex_loaded += self.dl.batch_size

    async def _produce_batch(self, batch_number: int):
        with hax.axis_mapping(self.mapping), self.dl.mesh:
            indices = range(batch_number * self.dl.batch_size, (batch_number + 1) * self.dl.batch_size, 1)

            # (begin, end) -> leaf index -> stacked array
            stacked_local_batch: dict[tuple[int, int], list[Array | hax.NamedArray]] = {}

            def get_local_batch(begin: int, end: int) -> list:
                key = (begin, end)
                if key in stacked_local_batch:
                    return stacked_local_batch[key]

                # TODO: if we ever do "big data" (i.e. huge examples) we might want to be able to load part of an example
                # which will require support from the datastore (i.e. tensorstore)
                individual_datums = blocking_wait(self.dl.data_store.get_batch(indices[begin:end]))

                device_batch = _stack_tree(self.dl.Batch.name, individual_datums)
                batch_leaves = hax.tree_util.tree_leaves(device_batch)

                stacked_local_batch[key] = batch_leaves

                return batch_leaves

            def get_local_data_for_leaf(indices: _TensorSliceIndex, leaf_index: int) -> Array | hax.NamedArray:
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
                        # TODO: this doesn't work with named axes
                        return leaf_data[(..., *other_indices)]

            def make_global_array_for_leaf(leaf_index, item_leaf_shape: ShapeSpec | NamedShapeSpec):
                raw_array = jax.make_array_from_callback(
                    to_raw_shape(item_leaf_shape),
                    jax.sharding.NamedSharding(self.dl.mesh, self._pspec_for(item_leaf_shape)),
                    lambda indices: get_local_data_for_leaf(indices, leaf_index),
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


def blocking_wait(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            result = [None]
            exception = [None]

            def run_coro():
                nonlocal result, exception
                try:
                    result[0] = asyncio.run(coro)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_coro)
            thread.start()
            thread.join()

            if exception[0]:
                raise exception[0]
            return result[0]
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
