import asyncio
import logging
from typing import Iterable, Iterator, Optional, TypeVar

import jax
from jax import Array
from jax.sharding import Mesh, PartitionSpec

import haliax as hax
from haliax import is_named_array
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
        data_store: AsyncDataset[Ex],
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
        self.data_store = data_store
        self.mesh = mesh
        self.Batch = Batch

        async def _exemplar_shape():
            x = await self.data_store.async_getitem(0)
            return _abstractify(x)

        self._exemplar_future = asyncio.create_task(_exemplar_shape())

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

        if self.dl.max_capacity is not None and self.dl.max_capacity >= 0:
            self._batches = self._produce_batches()
        else:
            self._batches = iter(BackgroundIterable(self._produce_batches, max_capacity=self.dl.max_capacity))

    def __next__(self):
        return next(self._batches)

    async def _produce_batches(self):
        _ex_leaves, _ex_structure = jax.tree_flatten(await self.dl._exemplar_future)

        batch_number = self._start_from_batch or 0
        total_ex_loaded = 0
        while True:
            if self.dl.data_store.is_finite():
                # wait until we're sure we have enough
                await self.dl.data_store.wait_until_len_at_least((batch_number + 1) * self.dl.batch_size)

            batch = self._produce_batch(batch_number)
            batch_number += 1
            yield batch

            total_ex_loaded += self.dl.batch_size

    async def _produce_batch(self, batch_number):
        indices = range(batch_number * self.dl.batch_size, batch_number + 1, self.dl.batch_size)

        # (begin, end) -> leaf index -> stacked array
        stacked_local_batch: dict[tuple[int, int], list[Array | hax.NamedArray]] = {}

        def get_local_batch(begin: int, end: int) -> list:
            key = (begin, end)
            if key in stacked_local_batch:
                return stacked_local_batch[key]

            # TODO: if we ever do "big data" (i.e. huge examples) we might want to be able to load part of an example
            # which will require support from the datastore (i.e. tensorstore)
            individual_datums = self.dl.data_store.get_batch(indices[begin:end])

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
            for leaf_index, item_leaf in enumerate(self._exemplar_leaves)
        ]

        gda_tree = jax.tree.unflatten(self._exemplar_shape, gda_leaves)

        return gda_tree

    def _pspec_for(self, shape_spec: ShapeSpec | NamedShapeSpec) -> PartitionSpec:
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            batch_name = hax.partitioning.physical_axis_name(self.dl.Batch, self.dl.axis_resources)
            return PartitionSpec(batch_name, *((None,) * (len(shape_spec.shape) - 1)))
        else:
            return hax.partitioning.pspec_for_axis(shape_spec.shape, self.axis_resources)  # type: ignore


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
