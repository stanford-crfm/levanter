import abc
from typing import Generic, TypeVar

from jax.random import PRNGKey


# from levanter.data import ShardCache

T = TypeVar("T")


class ItemSampler(Generic[T], abc.ABC):
    """
    Samples batches of data from a dataset.
    """

    # TODO: getstate/setstate

    @abc.abstractmethod
    def sample(self, index: int, *, key: PRNGKey) -> T:
        """
        Samples a batch of data from the dataset.

        Args:
            index: The index of the item to sample. This can be any nonnegative integer.
            key: The random key if you need additional randomness

        Returns:
            The sampled data.
        """
        raise NotImplementedError


class RowSampler(ItemSampler[T]):
    """
    Samples rows from a shard cache randomly.
    """

    def __init__(self, cache):
        self.cache = cache

    def sample(self, index, *, key: PRNGKey) -> T:
        max_index = self.cache.final_row_count()
        index = index % max_index

        return self.cache.get_row(index)
