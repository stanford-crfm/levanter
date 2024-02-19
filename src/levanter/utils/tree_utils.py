import dataclasses
from typing import TypeVar

import equinox as eqx
import jax
from jaxtyping import PyTree

from haliax.util import StringHolderEnum


T = TypeVar("T", bound=PyTree)


class NonePolicy(StringHolderEnum):
    PRESERVE = "preserve"
    REPLACE = "replace"
    ERROR = "error"


def inference_mode(tree: T, value: bool, none_policy: str = NonePolicy.REPLACE) -> T:
    """
    Analogous to [equinox.nn.inference_mode][] (neé [equinox.tree_inference][]), but
    it works in the presence of nones for the `inference` argument.
    """

    has_inference = lambda leaf: hasattr(leaf, "inference")  # noqa: E731

    def replace_fn(node):
        if not has_inference(node):
            return node

        if node.inference is None:
            if none_policy == NonePolicy.PRESERVE:
                return node
            elif none_policy == NonePolicy.ERROR:
                raise ValueError(f"None found in {tree}.inference with none_policy={none_policy}")
            else:
                assert none_policy == NonePolicy.REPLACE, f"Unknown none_policy {none_policy}"

        if dataclasses.is_dataclass(node):
            return dataclasses.replace(node, inference=value)
        else:
            return eqx.tree_at(lambda x: x.inference, node, value, is_leaf=lambda x: x is node)

    def rec_set(tree):
        if has_inference(tree):
            tree = replace_fn(tree)

        if jax.tree_util.tree_leaves(tree) == [tree]:
            return tree

        return jax.tree_util.tree_map(rec_set, tree, is_leaf=lambda x: has_inference(x) and tree is not x)

    return rec_set(tree)
