from typing import Dict, Iterator, Mapping, TypeVar

import jax.random
import numpy as np
from jaxtyping import PRNGKeyArray

from haliax.util import StringHolderEnum

from levanter.data import ShardableDataset


T = TypeVar("T")


class StopStrategy(metaclass=StringHolderEnum):
    FIRST_STOP_STRATEGY = "first_exhausted"
    ALL_STOP_STRATEGY = "all_exhausted"
    RESTART_STRATEGY = "restart"


class MixtureDataset(ShardableDataset[T]):
    """
    MixtureDataset supports loading data from multiple datasets. It takes a list of datasets and yields from them
    according to the weights.

    Args:
        datasets: A dict of datasets, where the key is the name of the dataset and the value is the dataset itself
        weights: weights for each dataset
        stop_strategy: strategy for stopping the iteration, by default RESTART_STRATEGY
            - FIRST_STOP_STRATEGY: stop when one dataset has been exhausted
            - ALL_STOP_STRATEGY: stop when all datasets have been exhausted
            - RESTART_STRATEGY: restart the dataset when it has been exhausted
        key: random key for datasets sampling
    """

    def __init__(
        self,
        datasets: Mapping[str, ShardableDataset[T]],
        weights: Dict[str, float],
        key: int | PRNGKeyArray,
        stop_strategy: str = StopStrategy.RESTART_STRATEGY,
    ):
        self.datasets = datasets
        self.weights = MixtureDataset._normalize_weights(weights)

        if stop_strategy not in StopStrategy:  # type: ignore
            raise ValueError(f"Stop strategy {stop_strategy} is not supported.")

        self.stop_strategy = stop_strategy

        if not isinstance(key, int):
            key = jax.random.randint(key, (), 0, 2**20).item()

        self.key = key

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]):
        """Normalize the weights to sum to 1"""
        total = sum(weights.values())
        if total == 0:
            raise ValueError(f"Datasets' weights cannot sum to 0, got {weights}")
        return {name: weight / total for name, weight in weights.items() if weight > 0}

    def shard(self, shard_id: int, num_shards: int) -> "MixtureDataset":
        """Return a MixtureDataset with the sharded datasets"""
        sharded = {name: dset.shard(shard_id, num_shards) for name, dset in self.datasets.items()}
        my_key = int(jax.random.randint(jax.random.PRNGKey(self.key), (num_shards,), 0, 2**20)[shard_id])
        return MixtureDataset(datasets=sharded, weights=self.weights, stop_strategy=self.stop_strategy, key=my_key)

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
                    case StopStrategy.RESTART_STRATEGY:
                        iterators[dataset_name] = iter(self.datasets[dataset_name])
                    case StopStrategy.FIRST_STOP_STRATEGY:
                        break
                    case StopStrategy.ALL_STOP_STRATEGY:
                        del iterators[dataset_name]
                        del current_weights[dataset_name]
                        if len(current_weights) == 0:
                            break
                        current_weights = self._normalize_weights(current_weights)
