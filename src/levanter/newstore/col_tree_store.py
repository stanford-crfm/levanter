import os
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree

from .jagged_array import JaggedArrayBuilder


# TODO at some point if we turn this into a real library, it would be nice to store the schema
# TODO: some data is probably best not stored as a jagged array, but as a flat array?
# TODO: also sometimes we might want a rowstore actually


def heuristic_is_leaf(x):
    if isinstance(x, list):
        return jnp.isscalar(x[0])
    else:
        return False


class TreeStoreBuilder:
    """
    A TreeStoreBuilder stores batched data as a tree of ragged arrays.
    """

    path: str
    mode: str
    tree: Optional[
        PyTree[JaggedArrayBuilder]
    ] = None  # This starts as None, but is set to a list of JaggedArrayBuilders

    def __init__(self, path: str, mode: str):
        self.path = path
        self.mode = mode

    @staticmethod
    def open(path: str, mode="a") -> "TreeStoreBuilder":
        """
        Open a TreeStoreBuilder from a file.
        """
        return TreeStoreBuilder(path, mode)

    def append_batch(self, batch: List[PyTree]):
        """
        Append a batch of data to the store.
        """

        if self.tree is None:

            def open_builder(tree_path, item):
                item = np.asarray(item)
                rank = item.ndim
                render_tree_path = "/".join(_render_path_elem(x) for x in tree_path)
                return JaggedArrayBuilder.open(
                    os.path.join(self.path, render_tree_path), mode=self.mode, item_rank=rank, dtype=item.dtype
                )

            self.tree = jtu.tree_map_with_path(open_builder, batch[0], is_leaf=heuristic_is_leaf)

        # TODO: I do wish zarr supported async
        jtu.tree_map(
            lambda writer, *xs: writer.extend([np.asarray(x) for x in xs]),
            self.tree,
            *batch,
            is_leaf=heuristic_is_leaf,
        )

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
