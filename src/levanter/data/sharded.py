import itertools
import jax
import numpy as np
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.interpreters.pxla import Mesh, PartitionSpec
from math import prod
from transformers import BatchEncoding
from typing import Iterator, Optional, Sequence, Tuple, TypeVar

import levanter.mesh
from haliax.partitioning import ResourceAxis
from levanter.data import Dataset
from levanter.data.dataset import ShardableDataset


In = TypeVar("In")
Ex = TypeVar("Ex")


# TODO: maybe generify this to work on more than just single sequence inputs
# TODO: write tests to verify this works when data spans multiple processes
# ExampleShape = Union[Tuple[int, ...], Sequence[Tuple[int, ...]]]

# This is hard for me to think about.

# We use GlobalDeviceArrays to coordinate data loading. A GlobalDeviceArray is an array that has a global shape
# but only has the data for some of the chunks of the array (namely, the ones on the local devices). Data can be
# replicated across devices, so that more than one device has the same chunk of data. This is useful for
# data or model parallelism.
# GlobalDeviceArrays are constructed using from_batched_callback, which passes in a list of slices that correspond to
# the entries in the GDA.
#
# For our purposes (data loading), we want to load batches of data from a dataset. We want to load the data in a
# way that is compatible with the GDA, so that we can use the GDA to coordinate the data loading.
# For standard language modeling tasks, the data is a sequence of tokens of some length, so the GDA has shape
# (batch_size, num_tokens). We want to load batches of data that are compatible with this shape.
# The device mesh is (data, model), and we want to replicate data across the model axis. We partition the above array as
# (data, None), meaning that the batch axis is distributed across the data axis. Each process is responsible for
# loading data for its devices. GDA's from_batched_callback will tell us how much data to load and for which device, so
# we just have to load it. We do need to make sure that, in the event the data axis is larger than
# num_devices_per_process, each process that is part of the same position in the device mesh loads the same data.


class ShardedIndexedDataset(Dataset[GlobalDeviceArray]):
    def __init__(
        self,
        token_dataset: ShardableDataset[BatchEncoding],
        mesh: Mesh,
        batch_size: int,
        *,
        override_process_data_pos: Optional[int] = None,  # for testing
        override_process_data_groups: Optional[int] = None,  # for testing
    ):
        self.mesh = mesh
        self.batch_size = batch_size

        process_data_pos = override_process_data_pos or levanter.mesh.process_mesh_position(mesh)[0]
        num_data_process_groups = override_process_data_groups or levanter.mesh.process_mesh_size(mesh)[0]

        if not override_process_data_groups:
            assert num_data_process_groups <= jax.process_count()

        self.token_dataset = token_dataset.shard(process_data_pos, num_data_process_groups)

    def __iter__(self) -> Iterator[GlobalDeviceArray]:
        # TODO: support not infinite iterators
        def loop_gen():
            while True:
                for ex in self.token_dataset:
                    yield ex

        it = loop_gen()

        batch_shape = self.batch_shape
        pspec = self.partition_spec

        assert len(batch_shape) == len(pspec)

        def callback(indices: Sequence[Tuple[slice, ...]]):
            # TODO: it seems like we may want to just directly index into the tokenized dataset somehow. This seems a
            #  bit more fragile
            # there is one entry in indices per device. They may be identical.
            # convert slices to tuples so we can use hashes
            out = []
            data_for_group = {}
            for index_group in indices:
                # begin, end, step
                my_indices: Tuple[Tuple[int, int, int], ...] = tuple(
                    s.indices(axis_size) for axis_size, s in zip(batch_shape, index_group)
                )
                assert all(s[2] == 1 for s in my_indices)  # ensure step is 1
                slice_sizes = [s[1] - s[0] for s in my_indices]
                num_examples = prod(slice_sizes[0:-1])
                if my_indices not in data_for_group:
                    data_for_group[my_indices] = np.stack(
                        list([ex["input_ids"] for ex in itertools.islice(it, num_examples)])
                    ).reshape(*slice_sizes)
                out.append(data_for_group[my_indices])

            return out

        while True:
            yield GlobalDeviceArray.from_batched_callback(
                batch_shape,
                self.mesh,
                pspec,
                callback,
            )

    @property
    def partition_spec(self):
        return PartitionSpec(ResourceAxis.DATA, None)

    @property
    def batch_shape(self):
        return (self.batch_size, self.token_dataset.seq_len)
