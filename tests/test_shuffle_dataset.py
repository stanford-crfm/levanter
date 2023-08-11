from typing import Iterator

from jax.random import PRNGKey

from levanter.data import Dataset, ShuffleDataset


class RangeDataset(Dataset[int]):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __iter__(self) -> Iterator[int]:
        yield from range(self.start, self.end)


def test_shuffle_dataset():
    dataset = RangeDataset(0, 100)
    assert list(dataset) == list(range(100))

    key = PRNGKey(0)
    shuffle_dataset = ShuffleDataset(dataset, key, 10)

    assert set(shuffle_dataset) == set(range(100))

    assert list(shuffle_dataset) != list(range(100))

    key2 = PRNGKey(2)
    shuffle_dataset2 = ShuffleDataset(dataset, key2, 10)
    assert list(shuffle_dataset2) != list(shuffle_dataset)
