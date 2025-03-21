from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PyTree

import haliax
import haliax as hax
from haliax.tree_util import scan_aware_tree_map

from levanter.utils.jax_utils import is_inexact_arrayish


def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return eqx.filter_jvp(eqx.filter_grad(f), (x,), (v,))[1]


def tree_gaussian_like(key, tree):
    """
    Samples a tree of gaussian noise with the same structure as `tree`, except for leaves which are not inexact arrays,
    for which it returns None
    """
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    rand_n = lambda x, key: jax.random.normal(key, x.shape) if is_inexact_arrayish(x) else None
    g = jax.tree_util.tree_map(rand_n, leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def map_flattened_linear_layers(
    f: Callable[[hax.nn.Linear], hax.nn.Linear],
    params: PyTree,
    *,
    or_else: Callable | None = None,
    is_leaf: Callable | None = None,
):
    """
    Apply a function to all Linear layers in a PyTree, flattening articulated input/output dims into single dims, then
    unflattening them back into the original structure. This method also takes care of vmapping over scan layers.

    The linear layers will be passed to the function `f` and the result will be used to replace the original linear layer.
    The linear layers passed to `f` will be flattened into 2D (named) arrays, and the result will be unflattened back into the original shape.
    The bias term, if any, will be passed as a 1D named arrays.
    The weight array will not be None, but the bias array may be None.

    Args:
        f: The function to apply to each Linear layer
        params: The PyTree of parameters
        or_else: optional function to apply to non-Linear leaves
        is_leaf: optional function to determine if a node is a leaf. Linears will always be considered leaves.

    Returns:
        The PyTree with the function applied to all Linear layers and the structure preserved otherwise.
        returned linear layers will be unfattened back to their original shape.

    """

    if is_leaf is None:
        is_leaf = lambda x: isinstance(x, hax.nn.Linear)
    else:
        _is_leaf = is_leaf
        is_leaf = lambda x: isinstance(x, hax.nn.Linear) or _is_leaf(x)

    def map_fn(p):
        if isinstance(p, hax.nn.Linear):
            if p.weight is None:
                return p
            return f(p)
        elif or_else is not None:
            return or_else(p)
        else:
            return p

    flattened_linear = haliax.state_dict.flatten_linear_layers(params)
    flattened_linear = scan_aware_tree_map(map_fn, flattened_linear, is_leaf=is_leaf)

    return haliax.state_dict.unflatten_linear_layers(params, flattened_linear)
