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
import levanter
from haliax import Axis
from haliax.partitioning import ResourceAxis
from levanter.data import ShardableDataset
from levanter.data.sharded import GlobalBatchDataset
from levanter.data.text import TokenizedDocumentCache, TokenSeqDataset
from levanter.shapes import NamedShapeSpec, ShapeSpec, shape_spec_of


def _small_dataset(seq_len=128) -> TokenSeqDataset:
    def token_iter():
        for i in range(200):
            yield BatchEncoding(
                {
                    "input_ids": np.tile(np.arange(seq_len, dtype=np.int32) + i * 1000, (1, 1)),
                }
            )

    cache = TokenizedDocumentCache.build_or_load(
        token_iter(),
        cache_dir=f"test_cache/{seq_len}",
        num_shards=128,
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
            check_batch_shard_consistency(dataset, batch)


def check_batch_shard_consistency(dataset: GlobalBatchDataset, batch):
    model_axis_size = dataset.mesh.devices.shape[1]
    for i, shard_i in enumerate(batch.global_shards):
        data_axis_pos_i = shard_i.device.id // model_axis_size
        model_axis_pos_i = shard_i.device.id % model_axis_size
        for j, shard_j in enumerate(batch.global_shards):
            data_axis_pos_j = shard_j.device.id // model_axis_size
            model_axis_pos_j = shard_j.device.id % model_axis_size

            item_shape = shape_spec_of(batch)

            data_is_sharded = any(q == ResourceAxis.DATA for q in dataset._pspec_for(item_shape))
            model_is_sharded = any(q == ResourceAxis.MODEL for q in dataset._pspec_for(item_shape))

            should_be_same = (not data_is_sharded or data_axis_pos_i == data_axis_pos_j) and (
                not model_is_sharded or model_axis_pos_i == model_axis_pos_j
            )

            if shard_i.data is not None and shard_j.data is not None:
                data_i = np.array(shard_i.data)
                data_j = np.array(shard_j.data)
                if should_be_same:
                    assert np.all(data_i == data_j)
                else:
                    assert not np.all(data_i == data_j)


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
            check_batch_shard_consistency(dataset, batch)


def test_sharded_data_loading_model_axis_1_override_process_indices():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh, haliax.axis_mapping({"batch": ResourceAxis.DATA}):
        datasets = []
        for process_index in range(2):
            seq_len = 128
            cache = _small_dataset(seq_len)
            Batch = Axis("batch", len(devices))
            dataset = GlobalBatchDataset(
                cache,
                mesh,
                Batch=Batch,
                override_process_data_pos=process_index,
                override_process_data_groups=2,
            )
            datasets.append(dataset)

        batches = [list(itertools.islice(dataset, 10)) for dataset in datasets]
        for (b1, b2) in zip(*batches):
            assert b1.shape == b2.shape
            assert jnp.all(b1._value != b2._value)
            shard_i: Shard
            check_batch_shard_consistency(dataset, b1)
            check_batch_shard_consistency(dataset, b2)


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
            check_structured_batch(dataset, batch)


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
            check_structured_batch(dataset, batch)


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
            check_structured_batch(dataset, batch)


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
            check_structured_batch(dataset, batch)


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
            check_structured_batch(dataset, batch)


def check_structured_batch(dataset: GlobalBatchDataset, batch):
    assert levanter.shapes.conforms(dataset.item_shape, batch)
    shard_i: Shard
    leaves = jax.tree_util.tree_leaves(batch)
    for leaf in leaves:
        check_batch_shard_consistency(dataset, leaf)
