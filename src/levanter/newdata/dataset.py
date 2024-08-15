import abc
import asyncio
from typing import Callable, Generic, Optional, Sequence, TypeVar

import jax.random
import numpy as np
from async_lru import alru_cache
from jax.random import PRNGKey

from levanter.newdata.prp import Permutation


T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
U = TypeVar("U")


class Dataset(abc.ABC, Generic[T_co]):
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns the final length of the data store.
        May raise if the length is not known.
        """

    @abc.abstractmethod
    def has_len(self) -> bool:
        """
        Whether the data store currently has a known length. If this returns False, then the length of the data store
        may change in the future.
        """
        pass

    @abc.abstractmethod
    def current_len(self) -> Optional[int]:
        """
        Returns the current length of the data store. If the length is infinite or not known, returns None.
        """
        pass

    def __getitem__(self, index: int) -> T_co:
        return self.get_batch([index])[0]

    @abc.abstractmethod
    def get_batch(self, indices: Sequence[int] | np.ndarray) -> Sequence[T_co]:
        pass

    def as_async_dataset(self) -> "AsyncDataset[T_co]":
        return WrappedAsyncDataset(self)


class AsyncDataset(abc.ABC, Generic[T_co]):
    @abc.abstractmethod
    async def async_len(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def length_is_known(self) -> bool:
        """Returns whether the length of the dataset is known.
        If this returns False, the current_len of the dataset may change in the future."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_finite(self) -> bool:
        """
        Returns whether the dataset will have a known length in the future (e.g. if it's being constructed).
        If this returns False, the length of the dataset is infinite or unknowable.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def current_len(self) -> Optional[int]:
        """
        Returns the current length of the dataset that won't require (expensive) waiting.

        If the current length is not known, returns None. This can block (TODO: should it?) if the length is not known
        yet but will be known in the future.
        """
        raise NotImplementedError

    async def async_getitem(self, index: int) -> T_co:
        return (await self.get_batch([index]))[0]

    @abc.abstractmethod
    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        raise NotImplementedError

    async def wait_until_len_at_least(self, length: int) -> int:
        """Returns the length of the dataset once it is at least `length` or if the dataset has a known length."""
        return await naive_busy_wait_until_len_at_least(self, length)

    def map(self, fn: Callable[[T_co], U]) -> "MappedAsyncDataset[U]":
        return MappedAsyncDataset(self, fn)


async def naive_busy_wait_until_len_at_least(dataset: AsyncDataset[T_co], length: int) -> int:
    """You should probably implement this in a more efficient way. This is just a naive implementation."""
    while not await dataset.length_is_known():
        current_len = await dataset.current_len()
        if current_len is None:
            raise ValueError("Dataset has unknown length")
        if current_len <= length:
            await asyncio.sleep(0.1)
        else:
            return current_len

    return await dataset.async_len()


class WrappedAsyncDataset(AsyncDataset[T_co]):
    def __init__(self, dataset: Dataset[T_co]):
        self.dataset = dataset

    async def async_len(self) -> int:
        return len(self.dataset)

    async def length_is_known(self) -> bool:
        return self.dataset.has_len()

    def is_finite(self) -> bool:
        return self.dataset.has_len()

    async def current_len(self) -> Optional[int]:
        return self.dataset.current_len()

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return self.dataset.get_batch(indices)

    async def async_getitem(self, index: int) -> T_co:
        return self.dataset[index]

    def __repr__(self):
        return f"WrappedAsyncDataset({repr(self.dataset)})"

    def __str__(self):
        return f"WrappedAsyncDataset({str(self.dataset)})"


class SequenceDataset(Dataset[T_co]):
    """Minimal implementation of a dataset that wraps a sequence. Mostly for testing purposes."""

    def __init__(self, data: Sequence[T_co]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def has_len(self) -> bool:
        return True

    def current_len(self) -> Optional[int]:
        return len(self.data)

    def __getitem__(self, idx: int) -> T_co:
        return self.data[idx]

    def get_batch(self, indices: Sequence[int] | np.ndarray) -> Sequence[T_co]:
        return [self.data[i] for i in indices]


class ListAsyncDataset(AsyncDataset[T]):
    """
    A simple dataset that wraps a list. Mostly for testing.
    """

    def __init__(self, data: list[T]):
        self.data = data
        self.is_complete = False
        self.complete_promise: asyncio.Future[None] = asyncio.Future()
        self.length_updated = asyncio.Condition()

    async def async_len(self) -> int:
        # this is the final length
        if not self.is_complete:
            await self.complete_promise
        return len(self.data)

    async def length_is_known(self) -> bool:
        return self.is_complete

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self.data)

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T]:
        await self.wait_until_len_at_least(max(indices) + 1)
        return [self.data[i] for i in indices]

    def append(self, item: T):
        self.data.append(item)
        asyncio.create_task(self.notify_length_update())

    def finalize(self):
        self.is_complete = True
        self.complete_promise.set_result(None)
        asyncio.create_task(self.notify_length_update())

    async def notify_length_update(self):
        async with self.length_updated:
            self.length_updated.notify_all()

    async def wait_until_len_at_least(self, length: int) -> int:
        if self.is_complete:
            return await self.async_len()

        async with self.length_updated:
            while len(self.data) < length and not self.is_complete:
                await self.length_updated.wait()

        return len(self.data)


class PermutationDataset(AsyncDataset[T_co]):
    """A permutation dataset that wraps another dataset and applies a permutation to the indices."""

    def __init__(self, dataset: AsyncDataset[T_co], permutation: Permutation):
        self.permutation = permutation
        self.dataset = dataset

    @staticmethod
    async def from_dataset(
        dataset: AsyncDataset[T_co] | Dataset[T_co], key: jax.random.PRNGKey
    ) -> "PermutationDataset[T_co]":
        if isinstance(dataset, AsyncDataset) and not dataset.is_finite():
            raise ValueError("PermutationDataset requires a dataset with an (eventual) known length")

        dataset = dataset if isinstance(dataset, AsyncDataset) else WrappedAsyncDataset(dataset)
        length = await dataset.async_len()
        return PermutationDataset(dataset, Permutation(length, key))

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    async def length_is_known(self) -> bool:
        return await self.dataset.length_is_known()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def current_len(self) -> Optional[int]:
        return await self.dataset.current_len()

    async def async_getitem(self, index: int) -> T_co:
        return await self.dataset.async_getitem(self.permutation(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return await self.dataset.get_batch([self.permutation(i) for i in indices])


class EraShufflingDataset(AsyncDataset[T_co]):
    """
    A dataset that shuffles the data in "eras" of fixed length. Era shuffling is somewhere in between a shuffle buffer
    and a permutation. It's a "local" permutation where pi(i) \in [ (i//L) * L, (i//L + 1) * L ) for some era length L.

    The advantages of era shuffling are:
    - It's stateless, so resumes are easy
    - Like shuffle buffers, it's a decent compromise between full shuffling and no shuffling
    - Like a shuffle buffer, it's streaming: we don't need to know the length of the data in advance

    The disadvantages are:
    - It's not as good as full shuffling
    - It distributes less well than a shuffle buffer does. It's more like a "local" shuffle buffer.
    - You have to wait for an era to fill before you can start shuffling it. With prefetching, this is less of an issue.


    # TODO: given the way tokenization works (where it runs way ahead of training), we can probably increase the era
    length # over time. This would be a nice feature to have.
    """

    def __init__(self, dataset: AsyncDataset[T_co], era_length: int, *, key: jax.random.PRNGKey):
        self.era_length = era_length
        self.dataset = dataset
        self.key = key

        @alru_cache(maxsize=4)  # we're mostly going to be going sequentially
        async def gen_era_permutation(era: int) -> Permutation:
            # edge case: final era may be shorter than era_length
            current_len = await self.dataset.wait_until_len_at_least((era + 1) * self.era_length)
            era_length = min(self.era_length, current_len - era * self.era_length)

            mix_key = jax.random.fold_in(key, era)
            return Permutation(era_length, mix_key)

        self.gen_era_permutation = gen_era_permutation

    async def _get_index(self, idx: int) -> int:
        if idx < 0:
            raise ValueError("Negative indices are not supported")
        era = idx // self.era_length
        permutation = await self.gen_era_permutation(era)
        return permutation(idx - era * self.era_length) + era * self.era_length

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    async def length_is_known(self) -> bool:
        return await self.dataset.length_is_known()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def current_len(self) -> Optional[int]:
        # nb this is the no-wait length, which means we might be a bit behind the length of the inner dataset
        inner_current_len = await self.dataset.current_len()
        if inner_current_len is None:
            return None

        # if we have the final length, and it's the inner_current_len, then we can return the final length
        if await self.length_is_known() and inner_current_len == await self.async_len():
            return inner_current_len

        # otherwise, we need to wait for the era to fill
        era = inner_current_len // self.era_length
        return era * self.era_length

    async def async_getitem(self, index: int) -> T_co:
        return await self.dataset.async_getitem(await self._get_index(index))

    async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
        return await self.dataset.get_batch([await self._get_index(i) for i in indices])

    def __repr__(self):
        return f"EraShufflingDataset({repr(self.dataset)}, era_length={self.era_length})"

    def __str__(self):
        return f"EraShufflingDataset({str(self.dataset)})"

    async def wait_until_len_at_least(self, length: int) -> int:
        # wait until we hit the next era
        next_era_end = (length // self.era_length + 1) * self.era_length
        return await self.dataset.wait_until_len_at_least(next_era_end)


class _Unspecified:
    pass


_UNSPECIFIED = _Unspecified()


class MappedAsyncDataset(AsyncDataset[U]):
    def __init__(
        self,
        dataset: AsyncDataset[T_co],
        fn: Callable[[T_co], U] | Callable[[T_co, Optional[PRNGKey]], U],
        *,
        key: Optional[PRNGKey] | _Unspecified = _UNSPECIFIED,
    ):
        self.dataset = dataset
        self.fn = fn
        self.key = key

    async def async_len(self) -> int:
        return await self.dataset.async_len()

    async def length_is_known(self) -> bool:
        return await self.dataset.length_is_known()

    def is_finite(self) -> bool:
        return self.dataset.is_finite()

    async def current_len(self) -> Optional[int]:
        return await self.dataset.current_len()

    async def async_getitem(self, index: int) -> U:
        if self.key is not _UNSPECIFIED:
            return self.fn(await self.dataset.async_getitem(index), self._maybe_fold_in_key(index))  # type: ignore
        return self.fn(await self.dataset.async_getitem(index))  # type: ignore

    def _maybe_fold_in_key(self, index):
        key = self.key
        if key is not None:
            key = jax.random.fold_in(self.key, index)
        return key

    async def get_batch(self, indices: Sequence[int]) -> Sequence[U]:
        if self.key is not _UNSPECIFIED:
            return [self.fn(await self.dataset.async_getitem(i), self._maybe_fold_in_key(i)) for i in indices]  # type: ignore
        return [self.fn(await self.dataset.async_getitem(i)) for i in indices]  # type: ignore

    async def wait_until_len_at_least(self, length: int) -> int:
        return await self.dataset.wait_until_len_at_least(length)
