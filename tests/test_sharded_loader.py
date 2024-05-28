import itertools
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import haliax as hax
from haliax import Axis
from haliax.partitioning import ResourceAxis

import levanter.data
from levanter.data.loader import ShardedBatchLoader, check_sharded_consistency
from test_utils import skip_if_not_enough_devices


NUM_SHARDS_TINY = 16


def _small_dataset(seq_len=128, num_sequences=200) -> levanter.data.ShardableDataset[Sequence[int]]:
    class SequenceDataset(levanter.data.ShardableDataset[np.ndarray]):
        def __init__(self, sequences: Sequence[np.ndarray]):
            self.sequences = sequences

        def shard(self, shard_idx: int, num_shards: int) -> levanter.data.ShardableDataset[np.ndarray]:
            return SequenceDataset(self.sequences[shard_idx::num_shards])

        def __iter__(self):
            yield from self.sequences

    # sequences = [list(range(i * 1000, i * 1000 + seq_len)) for i in range(num_sequences)]
    sequences = [np.arange(seq_len) + 1000 * i for i in range(num_sequences)]

    return SequenceDataset(sequences)


@skip_if_not_enough_devices(2)
def test_sharded_data_loading_model_axis_2():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        seq_len = 128
        cache = _small_dataset(seq_len)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(cache, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


def test_sharded_data_loading_model_axis_1():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        seq_len = 128
        cache = _small_dataset(seq_len)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(cache, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


class StructuredDataset(levanter.data.ShardableDataset):
    def __init__(self, seq_len, begin, end, stride):
        self.seq_len = seq_len
        self.begin = begin
        self.end = end
        self.stride = stride

    def __getitem__(self, item):
        return {
            "input_ids": np.arange(self.seq_len, dtype=np.int32) + item * 1000,
            "labels": np.arange(self.seq_len, dtype=np.int32) + item * 1000,
            "extra": {
                "input_ids": np.arange(self.seq_len, dtype=np.int32) + item * 1000,
                "mask": np.arange(self.seq_len * 2, dtype=np.int32).reshape(-1, 2) + item * 1000,
            },
        }

    def __iter__(self):
        for i in range(self.begin, self.end, self.stride):
            yield self[i]

    def shard(self, shard_id: int, num_shards: int):
        return StructuredDataset(self.seq_len, self.begin + shard_id, self.end, self.stride * num_shards)


def test_structured_batches_model_axis_1():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        seq_len = 128
        dataset = StructuredDataset(seq_len, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


class ScalarDataset(levanter.data.ShardableDataset[hax.NamedArray]):
    def __init__(self, begin, end, stride):
        self.begin = begin
        self.end = end
        self.stride = stride

    def __getitem__(self, item):
        return hax.named(jnp.array(item), ())

    def __iter__(self):
        for i in range(self.begin, self.end, self.stride):
            yield self[i]

    def shard(self, shard_id: int, num_shards: int):
        return ScalarDataset(self.begin + shard_id, self.end, self.stride * num_shards)


def test_can_batch_named_scalars():

    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        dataset = ScalarDataset(0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_2():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        seq_len = 128
        dataset = StructuredDataset(seq_len, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


class StructuredDatasetWithNames(levanter.data.ShardableDataset):
    def __init__(self, Height: Axis, Width: Axis, begin, end, stride):
        self.Height = Height
        self.Width = Width
        self.begin = begin
        self.end = end
        self.stride = stride

    def _gen_image(self, index):
        image = (
            np.arange(self.Height.size * self.Width.size, dtype=np.int32).reshape(self.Height.size, self.Width.size)
            + index * 1000
        )

        return hax.named(image, (self.Height, self.Width))

    def __getitem__(self, item):
        return {
            "input_ids": self._gen_image(item),
            "labels": self._gen_image(item),
            "extra": {
                "input_ids": self._gen_image(item),
                "mask": hax.arange(self.Height) + item * 1000,
            },
            "id": hax.named(jnp.array(item), ()),
        }

    def __iter__(self):
        for i in range(self.begin, self.end, self.stride):
            yield self[i]

    def shard(self, shard_id: int, num_shards: int):
        return StructuredDatasetWithNames(
            self.Height, self.Width, self.begin + shard_id, self.end, self.stride * num_shards
        )


def test_structured_batches_model_axis_1_with_names():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_2_with_names():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(4)
def test_structured_batches_model_axis_2_subsharded():
    """This tests data loading if individual datums are sharded too"""
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    Height = Axis("Height", 16)
    Width = Axis("Width", 16)
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA), Height.name: ResourceAxis.MODEL}):
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


def test_sharded_loader_doesnt_throw_away_data():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(1, -1, model_axis_size),
        (ResourceAxis.REPLICA, ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, hax.axis_mapping({"batch": (ResourceAxis.REPLICA, ResourceAxis.DATA)}):
        dataset = ScalarDataset(0, 256, 1)
        Batch = Axis("batch", len(devices))
        loader = ShardedBatchLoader(dataset, mesh, Batch)

        batches = list(itertools.islice(loader, 10))
        dataset_examples = list(itertools.islice(dataset, 10 * Batch.size))

        def unbatch_example(example):
            return example.unbind("batch")

        loader_examples = [ex for b in batches for ex in unbatch_example(b)]

        for ex_d, ex_l in zip(dataset_examples, loader_examples):
            assert jnp.all(ex_d.array == ex_l.array)
