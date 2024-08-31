import jax
import numpy as np
import pytest

from levanter.data.mixture import StopStrategy
from levanter.newdata import ListAsyncDataset, MixtureDataset


@pytest.fixture
def datasets():
    ds1 = ListAsyncDataset([1, 2, 3, 4, 5])
    ds2 = ListAsyncDataset([10, 20, 30, 40, 50])
    ds3 = ListAsyncDataset([100, 200, 300, 400, 500])
    ds1.finalize()
    ds2.finalize()
    ds3.finalize()
    return {"ds1": ds1, "ds2": ds2, "ds3": ds3}


@pytest.fixture
def weights():
    return {"ds1": 0.5, "ds2": 0.3, "ds3": 0.2}


@pytest.fixture
def block_size():
    return 10


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.mark.asyncio
async def test_mixture_dataset_getitem(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key, randomize_blocks=False)

    item = await mixture_ds.async_getitem(0)
    assert item in [1, 10, 100], f"Unexpected item: {item}"


@pytest.mark.asyncio
async def test_mixture_dataset_get_batch(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key, randomize_blocks=False)

    batch = await mixture_ds.get_batch([0, 1, 2])
    assert len(batch) == 3
    assert all(item in [1, 2, 3, 10, 20, 30, 100, 200, 300] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_block_assignments(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key)

    block_assignment = await mixture_ds._get_block(0)
    assert block_assignment is not None
    assert len(block_assignment) == block_size


@pytest.mark.skip
@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_first(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key, stop_strategy=StopStrategy.FIRST_STOP_STRATEGY)

    with pytest.raises(NotImplementedError):
        await mixture_ds.async_len()


@pytest.mark.asyncio
async def test_mixture_dataset_stop_strategy_restart(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key, stop_strategy=StopStrategy.RESTART_STRATEGY)

    with pytest.raises(ValueError):
        await mixture_ds.async_len()


@pytest.mark.asyncio
async def test_mixture_dataset_normalized_weights(datasets, key, block_size):
    weights = {"ds1": 0, "ds2": 0.5, "ds3": 0.5}
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key, randomize_blocks=False)

    batch = await mixture_ds.get_batch([0, 1, 2])
    assert len(batch) == 3
    assert all(item in [10, 20, 30, 100, 200, 300] for item in batch)


@pytest.mark.asyncio
async def test_mixture_dataset_unpermuted_ids(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key)

    unpermuted_ids = mixture_ds._compute_unpermuted_ids(mixture_ds._counts_per_block)
    assert len(unpermuted_ids) == block_size
    assert unpermuted_ids[0] >> 32 in range(3)  # Ensure the dataset ID is valid


@pytest.mark.asyncio
async def test_mixture_dataset_remap_indices(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key)

    remapped_indices = await mixture_ds._remap_indices(datasets["ds1"], [0, 1, 2])
    assert len(remapped_indices) == 3
    assert remapped_indices == [0, 1, 2]

    # check wrap around
    len_ds1 = await datasets["ds1"].async_len()
    remapped_indices = await mixture_ds._remap_indices(datasets["ds1"], [len_ds1 - 1, len_ds1, len_ds1 + 1])
    assert len(remapped_indices) == 3

    assert remapped_indices == [len_ds1 - 1, 0, 1]


@pytest.mark.asyncio
async def test_mixture_dataset_respects_weights(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key)

    # Check that the dataset respects the weights
    num_samples = 1000
    samples = await mixture_ds.get_batch(list(range(num_samples)))

    counts = {"ds1": 0, "ds2": 0, "ds3": 0}
    for sample in samples:
        if sample < 10:
            counts["ds1"] += 1
        elif sample < 100:
            counts["ds2"] += 1
        else:
            counts["ds3"] += 1

    for dataset, count in counts.items():
        assert abs(count / num_samples - weights[dataset]) < 0.1, f"Dataset {dataset} has unexpected weight"


@pytest.mark.asyncio
async def test_mixture_dataset_randomizes_blocks(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key)

    block_assignment_1 = await mixture_ds._get_block(0)
    block_assignment_2 = await mixture_ds._get_block(0)

    assert np.all(block_assignment_1 == block_assignment_2), "Block assignments should be randomized"

    block_assignment_3 = await mixture_ds._get_block(1)
    assert not np.all(block_assignment_1 == block_assignment_3), "Block assignments should be randomized"


@pytest.mark.asyncio
async def test_mixture_dataset_samples_all_elements(datasets, weights, block_size, key):
    mixture_ds = MixtureDataset(datasets, weights, block_size, key=key)

    num_samples = 1000
    samples = await mixture_ds.get_batch(list(range(num_samples)))

    assert len(samples) == num_samples
    assert set(samples) == {1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500}
