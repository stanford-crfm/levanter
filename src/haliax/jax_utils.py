import functools as ft
from typing import Any, Callable, List, Optional, Sequence, Union

import equinox as eqx
import jax
import numpy as np
from chex import PRNGKey
from equinox.module import Static
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import PyTree


def shaped_rng_split(key, split_shape: Union[int, Sequence[int]] = 2) -> jrandom.KeyArray:
    if isinstance(split_shape, int):
        num_splits = split_shape
        split_shape = (num_splits,) + key.shape
    else:
        num_splits = np.prod(split_shape)
        split_shape = tuple(split_shape) + key.shape

    if num_splits == 1:
        return jnp.reshape(key, split_shape)

    unshaped = maybe_rng_split(key, num_splits)
    return jnp.reshape(unshaped, split_shape)


def maybe_rng_split(key: Optional[PRNGKey], num: int = 2):
    """Splits a random key into multiple random keys. If the key is None, then it replicates the None. Also handles
    num == 1 case"""
    if key is None:
        return [None] * num
    elif num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    else:
        return jrandom.split(key, num)


def filter_eval_shape(fun: Callable, *args, **kwargs):
    """As `jax.eval_shape`, but allows any Python object as inputs and outputs"""

    # TODO: file a bug

    def _fn(_static, _dynamic):
        _args, _kwargs = eqx.combine(_static, _dynamic)
        _out = fun(*_args, **_kwargs)
        _dynamic_out, _static_out = eqx.partition(_out, is_jax_array_like)
        return _dynamic_out, Static(_static_out)

    dynamic, static = eqx.partition((args, kwargs), is_jax_array_like)
    dynamic_out, static_out = jax.eval_shape(ft.partial(_fn, static), dynamic)
    return eqx.combine(dynamic_out, static_out.value)


def filter_checkpoint(fun: Callable, *, prevent_cse: bool = True, policy: Optional[Callable[..., bool]] = None):
    """As `jax.checkpoint`, but allows any Python object as inputs and outputs"""

    @ft.wraps(fun)
    def _fn(_static, _dynamic):
        _args, _kwargs = eqx.combine(_static, _dynamic)
        _out = fun(*_args, **_kwargs)
        _dynamic_out, _static_out = eqx.partition(_out, is_jax_array_like)
        return _dynamic_out, Static(_static_out)

    checkpointed_fun = jax.checkpoint(_fn, prevent_cse=prevent_cse, policy=policy, static_argnums=(0,))

    @ft.wraps(fun)
    def wrapper(*args, **kwargs):
        dynamic, static = eqx.partition((args, kwargs), is_jax_array_like)
        dynamic_out, static_out = checkpointed_fun(static, dynamic)

        return eqx.combine(dynamic_out, static_out.value)

    return wrapper


def is_jax_array_like(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


# adapted from jax but exposed so i can use it
def broadcast_prefix(prefix_tree: Any, full_tree: Any, is_leaf: Optional[Callable[[Any], bool]] = None) -> List[Any]:
    """Broadcast a prefix tree to match the structure of a full tree."""
    result = []
    num_leaves = lambda t: jax.tree_util.tree_structure(t).num_leaves  # noqa: E731
    add_leaves = lambda x, subtree: result.extend([x] * num_leaves(subtree))  # noqa: E731
    jax.tree_util.tree_map(add_leaves, prefix_tree, full_tree, is_leaf=is_leaf)
    full_structure = jax.tree_util.tree_structure(full_tree)

    return jax.tree_util.tree_unflatten(full_structure, result)


def _is_none(x):
    return x is None


def _combine(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def combine(*pytrees: PyTree, is_leaf=None) -> PyTree:
    """Generalization of eqx.combine to support custom is_leaf functions

    **Returns:**

    A PyTree with the same structure as its inputs. Each leaf will be the first
    non-`None` leaf found in the corresponding leaves of `pytrees` as they are
    iterated over.
    """

    if is_leaf is None:
        is_leaf = _is_none
    else:
        _orig_is_leaf = is_leaf
        is_leaf = lambda x: _is_none(x) or _orig_is_leaf(x)  # noqa: E731

    return jax.tree_util.tree_map(_combine, *pytrees, is_leaf=is_leaf)


def _UNSPECIFIED():
    raise ValueError("unspecified")


def named_call(f=_UNSPECIFIED, name: Optional[str] = None):
    if f is _UNSPECIFIED:
        return lambda f: named_call(f, name)  # type: ignore
    else:
        if name is None:
            name = f.__name__
            if name == "__call__":
                if hasattr(f, "__self__"):
                    name = f.__self__.__class__.__name__  # type: ignore
                else:
                    name = f.__qualname__.rsplit(".", maxsplit=1)[0]  # type: ignore
            else:
                name = f.__qualname__

        return jax.named_scope(name)(f)
