import contextlib
import json
import warnings
from dataclasses import fields
from typing import Any, Callable, Optional, TypeVar

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree

from haliax.jax_utils import is_jax_array_like


X = TypeVar("X")


def jnp_to_python(a: jnp.ndarray):
    if isinstance(a, (float, int)):
        return float(a)
    elif a.shape == () or a.shape == (1,):
        return a.item()
    else:
        return a.tolist()


@contextlib.contextmanager
def use_cpu_device():
    """Temporarily sets the default device to CPU"""
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        yield


def is_inside_jit():
    """Returns True if we're currently inside a jit"""
    return isinstance(jnp.zeros(()), jax.core.Tracer)


def flops_estimate(fn, *args, **kwargs):
    """Estimates the flop count of a function"""
    return jax.jit(fn).lower(*args).cost_analysis()["flops"]


def parameter_count(model: PyTree):
    # especially with jax.vjp, we get duplicate arrays and want to uniq them
    # NB we need to use object identity here, mostly because of ShapedDtypeStruct
    leaves = {id(x): x for x in jax.tree_util.tree_leaves(model) if is_jax_array_like(x)}
    return sum(x.size for x in leaves.values())


_sync_counter = 0


def multihost_broadcast_sync(obj: X, is_source: Optional[bool] = None, timeout: float = 200.0) -> X:
    """
    Uses jax's unpublished distributed api to sync a value across hosts using json dump. If is_source is None, then
    process_index 0 is the source.
    """
    global _sync_counter
    key = f"LEVANTER_MULTIHOST_BROADCAST_SYNC{_sync_counter}"
    if is_source is None:
        is_source = jax.process_index() == 0

    if jax.process_count() == 1:
        return obj

    import jax._src.distributed as distributed
    from jaxlib.xla_extension import DistributedRuntimeClient

    client: Optional[DistributedRuntimeClient] = distributed.global_state.client

    if client is None:
        raise RuntimeError("multihost_broadcast_sync requires jax distributed client to be initialized")

    if is_source:
        # serialized = pickle.dumps(obj, 0)  # 0 is pickle protocol. jax only accepts utf-8, and 0 gives us ascii
        # client.key_value_set(key, serialized.decode("ascii"))
        serialized = json.dumps(obj)
        client.key_value_set(key, serialized)

    client.wait_at_barrier(f"multihost_broadcast_sync{_sync_counter}", timeout_in_ms=int(timeout * 1000.0))

    if not is_source:
        serialized = client.blocking_key_value_get(key, timeout_in_ms=int(timeout * 1000.0))
        obj = json.loads(serialized)

    _sync_counter += 1
    return obj


def wait_at_barrier(name: str, timeout: float = 200.0):
    """
    Uses jax's unpublished distributed api to wait at a barrier

    NB: the barrier names must be globally unique, so you should use a unique name for each barrier
    """
    import jax._src.distributed as distributed
    from jaxlib.xla_extension import DistributedRuntimeClient

    if jax.process_count() == 1:
        return

    client: Optional[DistributedRuntimeClient] = distributed.global_state.client

    if client is None:
        raise RuntimeError("multihost_broadcast_sync requires jax distributed client to be initialized")

    client.wait_at_barrier(name, timeout_in_ms=int(timeout * 1000.0))


# from https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
# python is a disgusting language
def _isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(isinstance(n, str) for n in f)


def leaf_key_paths(
    pytree,
    prefix: Optional[str] = "",
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
    use_state_dict_keys: bool = False,
):
    """Creates unique, hopefully meaningful key paths for each leaf in a pytree. This is useful for
    serialization mostly. This functions knows about dicts, lists, NamedTuples, tuples, and equinox-style modules"""
    # TODO: jax now has a tree_flatten_with_path function. We should use that instead
    rec = lambda x, p: leaf_key_paths(  # noqa: E731
        x, prefix=join_key(prefix, p), is_leaf=is_leaf, use_state_dict_keys=use_state_dict_keys
    )

    if is_leaf is not None and is_leaf(pytree):
        return prefix
    elif isinstance(pytree, dict):
        return {k: rec(v, k) for k, v in pytree.items()}
    elif _isnamedtupleinstance(pytree):
        d = {k: rec(v, k) for k, v in pytree._asdict().items()}
        return pytree.__class__(**d)
    elif isinstance(pytree, list):
        return [rec(v, str(i)) for i, v in enumerate(pytree)]
    elif isinstance(pytree, tuple):
        return tuple(rec(v, str(i)) for i, v in enumerate(pytree))
    elif isinstance(pytree, eqx.Module):
        names = []
        rec_values = []
        for field in fields(pytree):
            if field.metadata.get("static", False):
                continue
            field_name = field.name
            field = getattr(pytree, field_name)
            names.append(field_name)

            if use_state_dict_keys and hasattr(pytree, "_state_dict_key_map"):
                field_name = pytree._state_dict_key_map().get(field_name, field_name)

            rec_value = rec(field, field_name)
            rec_values.append(rec_value)

        _, tree_def = eqx.tree_flatten_one_level(pytree)
        out = jax.tree_util.tree_unflatten(tree_def, rec_values)
        return out
        # this doesn't work reliably because tree_at doesn't like none values
        # return eqx.tree_at(lambda m: [getattr(m, name) for name in names], pytree, rec_values, is_leaf=lambda x: x is None)
    else:
        leaves, treedef = jax.tree_util.tree_flatten(pytree, is_leaf=is_leaf)
        if len(leaves) == 1:
            return jax.tree_util.tree_unflatten(treedef, [f"{prefix}"])
        else:
            return jax.tree_util.tree_unflatten(treedef, [join_key(prefix, str(i)) for i in range(len(leaves))])


def join_key(prefix, k):
    if k is None:
        return prefix
    return f"{prefix}.{k}" if prefix else k


def key_iterator(key: PRNGKeyArray | int):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def is_inexact_arrayish(x):
    """
    Similar to [equinox.is_inexact_array][] but works on anything that has a shape and dtype
    and the dtype is inexact.

    Specifically, we want to work with [jax.ShapeDtypeStruct][]s, which are not arrays.
    """
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return jnp.issubdtype(x.dtype, jnp.inexact)
    else:
        return False


def tree_filter_like(template: X, tree: X) -> X:
    """
    Filters a tree to only include the leaves that are not None in the template.

    This is useful for filtering out nontrainable parameters from a tree.
    """

    def match_like(templ_leaf, tree_leaf):
        if templ_leaf is None:
            return None
        else:
            if tree_leaf is None:
                warnings.warn(f"Template has a non-None value where tree is None. Template value: {templ_leaf}")
            return tree_leaf

    return jax.tree_util.tree_map(match_like, template, tree, is_leaf=lambda x: x is None)


def as_arrayish(x):
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return x
    else:
        return jnp.asarray(x)
