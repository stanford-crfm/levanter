import itertools
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.global_device_array import Shard
from jax.experimental.maps import Mesh
from jaxtyping import PyTree
from transformers import BatchEncoding
from utils import skip_if_not_enough_devices

import haliax
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.data import ShardableDataset
from levanter.data.sharded import GlobalBatchDataset, check_sharded_consistency
from levanter.data.text import TokenizedDocumentCache, TokenSeqDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec


NUM_SHARDS_TINY = 16


def _small_dataset(seq_len=128, num_sequences=200) -> TokenSeqDataset:
    def token_iter():
        for i in range(num_sequences):
            yield BatchEncoding(
                {
                    "input_ids": np.tile(np.arange(seq_len, dtype=np.int32) + i * 1000, (1, 1)),
                }
            )

    cache = TokenizedDocumentCache.build_or_load(
        token_iter(),
        cache_dir=f"test_cache/{seq_len}",
        num_shards=NUM_SHARDS_TINY,
        flatten_docs=True,
    )

    return TokenSeqDataset(cache, seq_len)


@skip_if_not_enough_devices(2)
def test_sharded_data_loading_model_axis_2():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):

        seq_len = 128
        cache = _small_dataset(seq_len)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(cache, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            assert batch.shape == dataset.item_shape.shape
            shard_i: Shard
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


def test_sharded_data_loading_model_axis_1():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):

        seq_len = 128
        cache = _small_dataset(seq_len)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(cache, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            assert batch.shape == dataset.item_shape.shape
            shard_i: Shard
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


#
def test_sharded_data_loading_len_impact():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        cache = _small_dataset(64, num_sequences=NUM_SHARDS_TINY * 8)
        # 6400 tokens split across NUM_SHARDS_TINY shards
        Batch = Axis("batch", 8 * len(devices))
        process_1_len = len(
            GlobalBatchDataset(cache, mesh, Batch=Batch, override_process_data_pos=0, override_process_data_groups=1)
        )
        for process_count in [2, 4, 8]:
            dataset = GlobalBatchDataset(
                cache,
                mesh,
                Batch=Batch,
                override_process_data_pos=0,
                override_process_data_groups=process_count,
            )
            # we create this dataset with even numbers of shards, so we are guaranteed that the length won't change
            assert len(dataset) == process_1_len


class StructuredDataset(ShardableDataset):
    def __init__(self, seq_len, begin, end, stride):
        self.seq_len = seq_len
        self.begin = begin
        self.end = end
        self.stride = stride

    def __len__(self):
        return (self.end - self.begin) // self.stride

    def __getitem__(self, item):
        return {
            "input_ids": np.arange(self.seq_len, dtype=np.int32) + item * 1000,
            "labels": np.arange(self.seq_len, dtype=np.int32) + item * 1000,
            "extra": {
                "input_ids": np.arange(self.seq_len, dtype=np.int32) + item * 1000,
                "mask": np.arange(self.seq_len * 2, dtype=np.int32).reshape(-1, 2) + item * 1000,
            },
        }

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return {
            "input_ids": ShapeSpec((self.seq_len,), jnp.int32),
            "labels": ShapeSpec((self.seq_len,), jnp.int32),
            "extra": {
                "input_ids": ShapeSpec((self.seq_len,), jnp.int32),
                "mask": ShapeSpec((self.seq_len, 2), jnp.int32),
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
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        seq_len = 128
        dataset = StructuredDataset(seq_len, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(dataset, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_2():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        seq_len = 128
        dataset = StructuredDataset(seq_len, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(dataset, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


class StructuredDatasetWithNames(ShardableDataset):
    def __init__(self, Height: Axis, Width: Axis, begin, end, stride):
        self.Height = Height
        self.Width = Width
        self.begin = begin
        self.end = end
        self.stride = stride

    def __len__(self):
        return (self.end - self.begin) // self.stride

    def _gen_image(self, index):
        image = (
            np.arange(self.Height.size * self.Width.size, dtype=np.int32).reshape(self.Height.size, self.Width.size)
            + index * 1000
        )

        return haliax.named(image, (self.Height, self.Width))

    def __getitem__(self, item):
        return {
            "input_ids": self._gen_image(item),
            "labels": self._gen_image(item),
            "extra": {
                "input_ids": self._gen_image(item),
                "mask": haliax.arange(self.Height) + item * 1000,
            },
        }

    @property
    def item_shape(self) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
        return {
            "input_ids": NamedShapeSpec((self.Height, self.Width), jnp.int32),
            "labels": NamedShapeSpec((self.Height, self.Width), jnp.int32),
            "extra": {
                "input_ids": NamedShapeSpec((self.Height, self.Width), jnp.int32),
                "mask": NamedShapeSpec((self.Height,), jnp.int32),
            },
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
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(dataset, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(2)
def test_structured_batches_model_axis_2_with_names():
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        Height = Axis("Height", 16)
        Width = Axis("Width", 16)
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(dataset, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)


@skip_if_not_enough_devices(4)
def test_structured_batches_model_axis_2_subsharded():
    """This tests data loading if individual datums are sharded too"""
    devices = jax.devices()
    model_axis_size = 2

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    Height = Axis("Height", 16)
    Width = Axis("Width", 16)
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA, Height.name: ResourceAxis.MODEL}):
        dataset = StructuredDatasetWithNames(Height, Width, 0, 256, 1)
        Batch = Axis("batch", len(devices))
        dataset = GlobalBatchDataset(dataset, mesh, Batch)

        batches = list(itertools.islice(dataset, 10))
        for batch in batches:
            check_sharded_consistency(batch, check_disjoint_indices_are_different=True)
