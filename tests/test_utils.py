from functools import reduce
from typing import Callable, List, Optional, Sequence, TypeVar

import equinox as eqx
import jax
import pyarrow as pa
import pytest
from chex import assert_trees_all_close
from equinox import nn as nn
from equinox import static_field
from transformers import BatchEncoding

from levanter.data.shard_cache import BatchProcessor, ShardedDataSource
from levanter.data.text import _as_record_batch, _stack_batch_encodings


T = TypeVar("T")


def skip_if_not_enough_devices(count: int):
    return pytest.mark.skipif(len(jax.devices()) < count, reason=f"Not enough devices ({len(jax.devices())})")


class MLP(eqx.Module):
    """slightly less annoying MLP, used for testing purposes"""

    layers: List[nn.Linear]
    activation: Callable = eqx.static_field()
    final_activation: Callable = eqx.static_field()
    in_size: int = static_field()
    out_size: int = static_field()
    width_size: int = static_field()
    depth: int = static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(nn.Linear(in_size, out_size, key=keys[0]))
        else:
            layers.append(nn.Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(nn.Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(nn.Linear(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation  # type: ignore
        self.final_activation = final_activation  # type: ignore

    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None):
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


def assert_trees_not_close(a, b):
    try:
        assert_trees_all_close(jax.tree_util.tree_leaves(arrays_only(a)), jax.tree_util.tree_leaves(arrays_only(b)))
    except AssertionError:
        pass
    else:
        raise AssertionError("Trees are equal")


def arrays_only(x):
    return eqx.filter(x, eqx.is_inexact_array_like)


def has_torch():
    try:
        import torch  # noqa F401

        return True
    except ImportError:
        return False


def skip_if_no_torch(f):
    return pytest.mark.skipif(not has_torch(), reason="torch not installed")(f)


class IdentityProcessor(BatchProcessor[BatchEncoding]):
    def __call__(self, batch: Sequence[BatchEncoding]) -> pa.RecordBatch:
        stacked = reduce(_stack_batch_encodings, batch)
        return _as_record_batch(stacked)

    @property
    def num_cpus(self) -> int:
        return 0


class ShardsDataSource(ShardedDataSource[T]):
    def __init__(self, docs: List[List[T]]):
        self.docs = docs

    @property
    def shard_names(self) -> Sequence[str]:
        return [str(i) for i in range(len(self.docs))]

    def open_shard_at_row(self, shard_name: str, row: int):
        return self.docs[int(shard_name)][row:]


class SingleShardDocumentSource(ShardedDataSource[T]):
    def __init__(self, docs: List[T]):
        self.docs = docs

    @property
    def shard_names(self) -> Sequence[str]:
        return ["0"]

    def open_shard_at_row(self, shard_name: str, row: int):
        return self.docs[row:]
