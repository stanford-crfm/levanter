import asyncio
import os
from typing import Generic, List, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from haliax.jax_utils import is_jax_array_like

from .jagged_array import JaggedArrayStore


T = TypeVar("T", bound=PyTree)


# TODO at some point if we turn this into a real library, it would be nice to store the schema
# TODO: some data is probably best not stored as a jagged array, but as a flat array?
# TODO: also sometimes we might want a rowstore actually


def heuristic_is_leaf(x):
    if isinstance(x, list):
        return jnp.isscalar(x[0])
    else:
        return False


def heuristic_is_leaf_batched(x):
    if isinstance(x, list):
        return jnp.isscalar(x[0]) or is_jax_array_like(x[0])
    else:
        return False


class TreeStoreBuilder(Generic[T]):
    """
    A TreeStoreBuilder stores batched data as a tree of ragged arrays.

    This class could implement MutableSequence, but that would probably give the wrong impression.
    In particular, a slice returns a T with JaggedArrays as leaves, rather than a list of Ts.
    """

    path: str
    mode: str
    tree: PyTree[JaggedArrayStore]

    def __init__(self, tree, path: str, mode: str):
        self.path = path
        self.mode = mode
        self.tree = tree

    @staticmethod
    def open(exemplar: T, path: str, *, mode="a") -> "TreeStoreBuilder":
        """
        Open a TreeStoreBuilder from a file.
        """
        tree = _construct_builder_tree(exemplar, path, mode)
        return TreeStoreBuilder(tree, path, mode)

    def append(self, ex: T):
        return self.extend([ex])

    def extend(self, batch: List[T]):
        """
        Append a batch of data to the store.
        """
        # TODO: I do wish zarr supported async
        jtu.tree_map(
            lambda writer, *xs: writer.extend([np.asarray(x) for x in xs]),
            self.tree,
            *batch,
            is_leaf=heuristic_is_leaf,
        )

    def extend_with_batch(self, batch: T):
        """
        Append a batch of data (as a pytree with batched leaves) to the store.

        This method works only when the "leaves" are lists of numpy arrays or scalars.
        For instance, HF's BatchEncoding is a dict of lists of numpy arrays.
        """
        jtu.tree_map(
            lambda writer, xs: writer.extend([np.asarray(x) for x in xs]),
            self.tree,
            batch,
            is_leaf=heuristic_is_leaf_batched,
        )

    async def extend_with_batch_async(self, batch: T):
        """
        Append a batch of data (as a pytree with batched leaves) to the store.

        This method works only when the "leaves" are lists of numpy arrays or scalars.
        For instance, HF's BatchEncoding is a dict of lists of numpy arrays.
        """
        futures = jtu.tree_map(
            lambda writer, xs: writer.extend_async([np.asarray(x) for x in xs]),
            self.tree,
            batch,
            is_leaf=heuristic_is_leaf_batched,
        )

        await asyncio.gather(*jax.tree_leaves(futures))

    def trim_to_size(self, size: int):
        """
        Trim the store to a given size.
        """
        self.tree = jtu.tree_map(lambda writer: writer.trim_to_size(size), self.tree, is_leaf=heuristic_is_leaf)

    async def trim_to_size_async(self, size: int):
        """
        Trim the store to a given size.
        """
        futures = jtu.tree_map(lambda writer: writer.trim_to_size_async(size), self.tree, is_leaf=heuristic_is_leaf)
        leaves, structure = jax.tree_flatten(futures)

        awaitted = await asyncio.gather(*leaves)
        self.tree = jax.tree_unflatten(structure, awaitted)

    def reload(self) -> "TreeStoreBuilder":
        """
        Close the builder and return a TreeStore.
        """
        tree = jtu.tree_map(lambda builder: builder.reload(), self.tree, is_leaf=heuristic_is_leaf)
        return TreeStoreBuilder(tree, self.path, self.mode)

    def __len__(self):
        if self.tree is None:
            return 0
        else:
            return len(jax.tree.leaves(self.tree)[0])

    def __getitem__(self, item):
        if self.tree is None:
            raise IndexError("No data in store")
        else:
            return jtu.tree_map(lambda reader: reader[item], self.tree, is_leaf=heuristic_is_leaf)

    def __iter__(self):
        if self.tree is None:
            return
        else:
            for i in range(len(self)):
                yield self[i]


def _construct_builder_tree(exemplar, path, mode):
    def open_builder(tree_path, item):
        item = np.asarray(item)
        rank = item.ndim
        render_tree_path = "/".join(_render_path_elem(x) for x in tree_path)
        return JaggedArrayStore.open(os.path.join(path, render_tree_path), mode=mode, item_rank=rank, dtype=item.dtype)

    return jtu.tree_map_with_path(open_builder, exemplar, is_leaf=heuristic_is_leaf)


def _render_path_elem(x):
    match x:
        case jtu.DictKey(key):
            return f"{key}"
        case jtu.GetAttrKey(key):
            return f"{key}"
        case jtu.SequenceKey(i):
            return f"{i}"
        case jtu.FlattenedIndexKey(i):
            return f"{i}"
        case _:
            return str(x)


# class TokenSeqDataset:
#     """
#     A dataset of sequences of tokens of fixed length, materialized from a collection of JaggedArrayStores,
#     which have typically much longer sequences. This class takes consecutive sequences of tokens from the builders
#     and slices/concats them to form the dataset.
#     """
#
#     def __init__(
#         self, token_arrays: Sequence[JaggedArrayStore], token_counts: Sequence[int], seq_len: int, pad_token: int
#     ):
#         self.token_arrays = token_arrays
#
#         def _round_to_nearest_multiple(x, y):
#             return x + y - x % y
#
#         token_counts_padded = np.array([_round_to_nearest_multiple(x, seq_len) for x in token_counts])
#         seq_counts = token_counts_padded // seq_len
#         self.seq_counts_cumsum = np.concatenate([np.asarray([0]), np.cumsum(seq_counts)])
#
#         self.seq_len = seq_len
#         self.pad_token = pad_token
#
#     def __len__(self):
#         return self.seq_counts_cumsum[-1]
#
#     def __getitem__(self, seq_id):
#         return asyncio.run(self.get_item_async(seq_id))
#
#     async def get_item_async(self, seq_id):
#         # TODO: accept slices and such?
#         shard_id = np.searchsorted(self.seq_counts_cumsum, seq_id, side="right") - 1
#         shard_start = self.seq_counts_cumsum[shard_id]
#         shard_end = self.seq_counts_cumsum[shard_id + 1]
#         shard_seq_id = seq_id - shard_start
#
#         shard_seq_start = shard_seq_id * self.seq_len
#         shard_seq_end = min((shard_seq_id + 1) * self.seq_len, self.token_arrays[shard_id].data_size)
#
#         shard_seq = await self.token_arrays[shard_id].data[shard_seq_start:shard_seq_end].read()
#         pad_len = self.seq_len - (shard_seq_end - shard_seq_start)
#         padded_seq = np.concatenate([shard_seq, np.full(pad_len, self.pad_token, dtype=shard_seq.dtype)])
#
#         return padded_seq