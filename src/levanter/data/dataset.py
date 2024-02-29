from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, TypeVar

import jax.random as jrandom
from jax.random import PRNGKey


T = TypeVar("T", covariant=True)


class Dataset(Iterable[T], ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError


class ShardableDataset(Dataset[T], ABC):
    @abstractmethod
    def shard(self, shard_id: int, num_shards: int) -> "ShardableDataset[T]":
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError


class InMemoryDataset(ShardableDataset[T]):
    def __init__(self, items: List[T]):
        self.items = items

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def shard(self, shard_id: int, num_shards: int) -> "InMemoryDataset[T]":
        return InMemoryDataset(self.items[shard_id::num_shards])


class ShuffleDataset(ShardableDataset[T]):
    def __init__(self, dataset: Dataset[T], key: PRNGKey, buffer_size: int):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.key = key

    def shard(self, shard_id: int, num_shards: int) -> "ShuffleDataset":
        key = jrandom.fold_in(self.key, shard_id)
        return ShuffleDataset(self.dataset.shard(shard_id, num_shards), key, self.buffer_size)  # type: ignore

    def __iter__(self) -> Iterator[T]:
        inner = iter(self.dataset)
        buffer: List[T] = []
        current_key = self.key

        for item in inner:
            if len(buffer) == self.buffer_size:
                current_key, subkey = jrandom.split(current_key)
                i = jrandom.randint(subkey, (), 0, len(buffer))
                yield buffer[i]
                buffer[i] = item
            else:
                buffer.append(item)

        while len(buffer) > 0:
            current_key, subkey = jrandom.split(current_key)
            i = jrandom.randint(subkey, (), 0, len(buffer))
            yield buffer[i]
            del buffer[i]
