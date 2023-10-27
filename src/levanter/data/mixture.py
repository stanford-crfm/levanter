from typing import Dict, Iterator, Mapping, TypeVar

import jax.random
import numpy as np
from jax.random import PRNGKey
from jaxtyping import PRNGKeyArray

from haliax.util import StringHolderEnum

from levanter.data import ShardableDataset


T = TypeVar("T")


class StopStrategy(metaclass=StringHolderEnum):
    FIRST_STOP_STRATEGY = "first_exhausted"
    ALL_STOP_STRATEGY = "all_exhausted"


class MixtureDataset(ShardableDataset[T]):
    """
    MixtureDataset supports loading data from multiple datasets. It takes a list of datasets and yields from them
    according to the weights.

    Args:
        datasets: A dict of datasets, where the key is the name of the dataset and the value is the dataset itself
        weights: weights for each dataset
        stop_strategy: strategy for stopping the iteration, by default FIRST_STOP_STRATEGY
            - FIRST_STOP_STRATEGY: stop when one dataset has been exhausted
            - ALL_STOP_STRATEGY: stop when all datasets have been exhausted
        key: random key for datasets sampling
    """

    def __init__(
        self,
        datasets: Mapping[str, ShardableDataset[T]],
        weights: Dict[str, float],
        stop_strategy: str = StopStrategy.FIRST_STOP_STRATEGY,
        key: int | PRNGKeyArray = 0,
    ):
        self.datasets = datasets
        self.weights = MixtureDataset._normalize_weights(weights)

        if stop_strategy not in [StopStrategy.FIRST_STOP_STRATEGY, StopStrategy.ALL_STOP_STRATEGY]:
            raise ValueError(f"Stop strategy {stop_strategy} is not supported.")

        self.stop_strategy = stop_strategy

        if not isinstance(key, int):
            key = jax.random.randint(PRNGKey(key)[0], (), 0, 2**31).item()

        self.key = key

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]):
        """Normalize the weights to sum to 1"""
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Datasets' Weights cannot sum to 0")
        return {name: weight / total for name, weight in weights.items() if weight > 0}

    def shard(self, shard_id: int, num_shards: int) -> "MixtureDataset":
        """Return a MixtureDataset with the sharded datasets"""
        sharded = {name: dset.shard(shard_id, num_shards) for name, dset in self.datasets.items()}
        return MixtureDataset(sharded, self.weights)

    def __iter__(self) -> Iterator[np.ndarray]:
        iterators = {name: iter(dataset) for name, dataset in self.datasets.items()}
        current_weights = self._normalize_weights(self.weights)
        rng = np.random.default_rng(self.key)

        while True:
            dataset_name = rng.choice(list(current_weights.keys()), p=list(current_weights.values()))
            try:
                item = next(iterators[dataset_name])
                yield item
            except StopIteration:
                match self.stop_strategy:
                    case StopStrategy.FIRST_STOP_STRATEGY:
                        break
                    case StopStrategy.ALL_STOP_STRATEGY:
                        del iterators[dataset_name]
                        del current_weights[dataset_name]
                        if len(current_weights) == 0:
                            break
                        current_weights = self._normalize_weights(current_weights)
