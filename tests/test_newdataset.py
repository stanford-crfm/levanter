import asyncio

import jax.random
import pytest

from levanter.data import EraShufflingDataset, PermutationDataset
from levanter.data._prp import Permutation
from levanter.data.dataset import AsyncifiedDataset, ListAsyncDataset, SequenceDataset


@pytest.mark.asyncio
async def test_length_of_sequence_dataset_is_accurate():
    data = [1, 2, 3]
    dataset = SequenceDataset(data)
    assert len(dataset) == 3


@pytest.mark.asyncio
async def test_sequence_dataset_get_item_returns_correct_item():
    data = ["a", "b", "c"]
    dataset = SequenceDataset(data)
    assert dataset[1] == "b"


@pytest.mark.asyncio
async def test_wrapped_async_dataset_async_len_matches_sync_len():
    data = [1, 2, 3]
    dataset = SequenceDataset(data)
    wrapped_dataset = AsyncifiedDataset(dataset)
    assert await wrapped_dataset.async_len() == 3


@pytest.mark.asyncio
async def test_list_async_dataset_appends_and_finalizes_correctly():
    dataset = ListAsyncDataset([])
    dataset.append("a")
    dataset.finalize()
    assert await dataset.async_len() == 1
    assert await dataset.get_batch([0]) == ["a"]


@pytest.mark.asyncio
async def test_permutation_dataset_is_at_least_sometimes_permuted():
    for seed in range(10):
        data = [1, 2, 3, 4]
        dataset = SequenceDataset(data)
        wrapped_dataset = AsyncifiedDataset(dataset)
        permutation = Permutation(4, jax.random.PRNGKey(seed))
        permuted_dataset = PermutationDataset(wrapped_dataset, permutation)
        if await permuted_dataset.get_batch([0, 1, 2, 3]) != [1, 2, 3, 4]:
            return

    pytest.fail("PermutationDataset did not permute the data")


@pytest.mark.asyncio
async def test_era_shuffling_dataset_returns_correct_length():
    data = list(range(100))
    dataset = ListAsyncDataset(data)
    era_length = 10
    key = jax.random.PRNGKey(0)
    shuffling_dataset = EraShufflingDataset(dataset, era_length, key=key)
    assert await shuffling_dataset.current_len() == 100
    assert not await shuffling_dataset.final_length_is_known()

    dataset.append(1)
    assert await shuffling_dataset.current_len() == 100


@pytest.mark.asyncio
async def test_era_shuffling_dataset_get_batch_returns_shuffled_batch():
    data = list(range(20))
    dataset = ListAsyncDataset(data)
    dataset.finalize()
    era_length = 5
    key = jax.random.PRNGKey(0)
    shuffling_dataset = EraShufflingDataset(dataset, era_length, key=key)
    batch_indices = [0, 1, 2, 3, 4]
    batch = await shuffling_dataset.get_batch(batch_indices)
    assert set(batch) == set([0, 1, 2, 3, 4])  # Ensures all elements are from the first era but does not assume order
    assert batch != [0, 1, 2, 3, 4]  # Ensures the batch is shuffled


@pytest.mark.asyncio
async def test_era_shuffling_can_grow():
    data = list(range(5))
    dataset = ListAsyncDataset(data)
    era_length = 5
    key = jax.random.PRNGKey(0)
    shuffling_dataset = EraShufflingDataset(dataset, era_length, key=key)
    batch_indices = [0, 1, 2, 3, 4]
    batch = await shuffling_dataset.get_batch(batch_indices)
    assert set(batch) == set([0, 1, 2, 3, 4])

    for i in range(5):
        dataset.append(i + 5)

    assert await shuffling_dataset.current_len() == 10
    assert not await shuffling_dataset.final_length_is_known()
    batch = await shuffling_dataset.get_batch(list(range(10)))

    assert set(batch) == set(range(10))
    assert set(batch[0:5]) == set([0, 1, 2, 3, 4])
    assert set(batch[5:10]) == set([5, 6, 7, 8, 9])

    # now make sure that we can await data and it does get fulfilled
    # this should timeout if we try to await it
    coro = dataset.get_batch([11])
    try:
        await asyncio.wait_for(coro, timeout=0.1)
        pytest.fail("Should have timed out")
    except asyncio.TimeoutError:
        pass

    async def append_data():
        await asyncio.sleep(0.1)
        for i in range(10, 15):
            dataset.append(i)

    coro = dataset.getitem_async(11)

    _, r = await asyncio.gather(append_data(), coro)
    assert r in range(10, 15)

    coro2 = shuffling_dataset.wait_until_len_at_least(20)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(coro2, timeout=0.1)

    assert await shuffling_dataset.current_len() == 15

    coro2 = shuffling_dataset.wait_until_len_at_least(20)
    dataset.append(15)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(coro2, timeout=0.1)

    assert await shuffling_dataset.current_len() == 15

    coro2 = shuffling_dataset.wait_until_len_at_least(20)
    dataset.finalize()
    await asyncio.wait_for(coro2, timeout=0.1)

    assert await dataset.async_len() == 16
    assert await shuffling_dataset.current_len() == 16

    coro = shuffling_dataset.get_batch(list(range(16)))

    batch = await coro
    assert set(batch) == set(range(16))
