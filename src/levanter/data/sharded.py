import itertools
from typing import Iterator, Optional, Sequence, Tuple, TypeVar, Union

import jax
import numpy as np
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.interpreters.pxla import Mesh, PartitionSpec
from jaxtyping import PyTree

import levanter.mesh
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.data import Dataset
from levanter.data.dataset import ShardableDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec


In = TypeVar("In")
Ex = TypeVar("Ex")


# TODO: maybe generify this to work on more than just single sequence inputs
# TODO: write tests to verify this works when data spans multiple processes
# ExampleShape = Union[Tuple[int, ...], Sequence[Tuple[int, ...]]]

_TensorSliceIndex = Tuple[slice, ...]


class GlobalBatchDataset(Dataset[GlobalDeviceArray]):
    """
    GlobalBatchDataset wraps a "local dataset" (a dataset that is shardable and can be iterated over) to produce
    GlobalDeviceArrays representing batches of data. A GlobalDeviceArray is an array that has a global shape
    but only has the data for some of the chunks of the array (namely, the ones on the local devices).
    Thus, each process loads the data for its devices.

    The details are a bit complex: We have a device mesh of shape (data, model). We want each row of the device mesh to
    get batch_size//num_rows examples. Usually, a process will be responsible for one or more entire rows, meaning
    that it wil load data that is distinct from every other process. However, if num_cols > num_devices_per_process,
    then some processes will need to load the same data. We use the process_mesh_position to determine which data to
    load, by determining which row(s) of the device mesh the process is responsible for.

    For now GlobalBatchDataset is restricted to datasets that return a single sequence of tokens.

    :arg local_dataset: a dataset that is shardable and can be iterated over
    :arg mesh: the device mesh
    :arg Batch: the batch size
    """

    def __init__(
        self,
        local_dataset: ShardableDataset[Sequence[int]],
        mesh: Mesh,
        Batch: Axis,
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

        self.local_dataset = local_dataset.shard(process_data_pos, num_data_process_groups)

    def __iter__(self) -> Iterator[GlobalDeviceArray]:
        # TODO: support not infinite iterators
        def loop_gen():
            while True:
                for ex in self.local_dataset:
                    yield ex

        it = loop_gen()

        item_shape = self.item_shape.shape
        pspec = self.partition_spec

        assert len(item_shape) == len(pspec)

        # This callback takes a sequence of slices indicating a grid of the data to load, and returns the data
        # We're mostly going to ignore the slices, because we're streaming the data
        # We get one slice per device: the slices will be identical if the data is replicated
        # TODO: we may want to just directly index into the tokenized dataset somehow. This seems a bit more fragile

        def callback(indices: Sequence[_TensorSliceIndex]):
            out = []

            # because more than one device can get the same data, we need to make sure we only load it once since we're
            # streaming. This is the cache
            data_for_slice = {}

            for tslice_index in indices:
                begin_ends_for_index = self._get_begin_end_for_slice(item_shape, tslice_index)
                slice_sizes = [s[1] - s[0] for s in begin_ends_for_index]

                num_examples = slice_sizes[0]

                if begin_ends_for_index not in data_for_slice:
                    data_for_slice[begin_ends_for_index] = np.stack(
                        list([ex for ex in itertools.islice(it, num_examples)])
                    )
                out.append(data_for_slice[begin_ends_for_index])

            return out

        while True:
            yield GlobalDeviceArray.from_batched_callback(
                item_shape,
                self.mesh,
                pspec,
                callback,
            )

    @staticmethod
    def _get_begin_end_for_slice(tensor_shape, tslice_index) -> Tuple[Tuple[int, int], ...]:
        # begin, end, step
        my_indices: Tuple[Tuple[int, int, int], ...] = tuple(
            s.indices(axis_size) for axis_size, s in zip(tensor_shape, tslice_index)
        )
        assert all(s[2] == 1 for s in my_indices)  # ensure step is 1
        return tuple(s[0:2] for s in my_indices)

    @property
    def partition_spec(self):
        return PartitionSpec(ResourceAxis.DATA, None)

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        def _batchify_shape_spec(shape_spec: Union[ShapeSpec, NamedShapeSpec]):
            shape = shape_spec.shape
            assert shape is not None, "item_shape must have a fully determined shape to work with batching"
            if isinstance(shape_spec, NamedShapeSpec):
                return NamedShapeSpec((self.Batch,) + shape, shape_spec.dtype)
            else:
                return ShapeSpec((self.Batch.size,) + shape, shape_spec.dtype)

        return jax.tree_map(_batchify_shape_spec, self.local_dataset.item_shape)
