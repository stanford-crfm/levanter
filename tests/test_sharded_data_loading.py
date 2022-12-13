import itertools
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.global_device_array import GlobalDeviceArray, Shard
from jax.experimental.maps import Mesh
from transformers import BatchEncoding
from utils import skip_if_not_enough_devices

from haliax.partitioning import ResourceAxis
from levanter.data.sharded import ShardedIndexedDataset
from levanter.data.text import TokenizedDocumentCache, TokenSeqDataset


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
    with mesh:

        seq_len = 128
        cache = _small_dataset(seq_len)
        dataset = ShardedIndexedDataset(cache, mesh, batch_size=len(devices))

        batches: List[GlobalDeviceArray] = list(itertools.islice(dataset, 10))
        for batch in batches:
            assert batch.shape == dataset.batch_shape
            shard_i: Shard
            check_batch_shard_consistency(batch, mesh)


def check_batch_shard_consistency(batch, mesh):
    model_axis_size = mesh.devices.shape[1]
    for i, shard_i in enumerate(batch.global_shards):
        data_axis_pos_i = shard_i.device.id // model_axis_size
        for j, shard_j in enumerate(batch.global_shards):
            data_axis_pos_j = shard_j.device.id // model_axis_size
            if shard_i.data is not None and shard_j.data is not None:
                if data_axis_pos_i == data_axis_pos_j:
                    assert np.all(np.array(shard_i.data) == np.array(shard_j.data))
                else:
                    assert not np.all(np.array(shard_i.data) == np.array(shard_j.data))


def test_sharded_data_loading_model_axis_1():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh:

        seq_len = 128
        cache = _small_dataset(seq_len)
        dataset = ShardedIndexedDataset(cache, mesh, batch_size=len(devices))

        batches: List[GlobalDeviceArray] = list(itertools.islice(dataset, 10))
        for batch in batches:
            assert batch.shape == dataset.batch_shape
            shard_i: Shard
            check_batch_shard_consistency(batch, mesh)


def test_sharded_data_loading_model_axis_1_override_process_indices():
    devices = jax.devices()
    model_axis_size = 1

    mesh = Mesh(
        np.array(devices).reshape(-1, model_axis_size),
        (ResourceAxis.DATA, ResourceAxis.MODEL),
    )
    with mesh:
        datasets = []
        for process_index in range(2):
            seq_len = 128
            cache = _small_dataset(seq_len)
            dataset = ShardedIndexedDataset(
                cache,
                mesh,
                batch_size=len(devices),
                override_process_data_pos=process_index,
                override_process_data_groups=2,
            )
            datasets.append(dataset)

        batches: List[List[GlobalDeviceArray]] = [list(itertools.islice(dataset, 10)) for dataset in datasets]
        b1: GlobalDeviceArray
        for (b1, b2) in zip(*batches):
            assert b1.shape == b2.shape
            assert jnp.all(b1._value != b2._value)
            shard_i: Shard
            check_batch_shard_consistency(b1, mesh)
            check_batch_shard_consistency(b2, mesh)
