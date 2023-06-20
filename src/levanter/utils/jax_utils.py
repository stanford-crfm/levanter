import functools
import functools as ft
import json
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar

import equinox as eqx
import jax
from chex import PRNGKey
from jax import lax
from jax import numpy as jnp
from jax import random as jrandom
from jax.sharding import PartitionSpec
from jaxtyping import PyTree

from haliax.jax_utils import is_jax_array_like, shaped_rng_split
from haliax.util import ensure_tuple


def jnp_to_python(a: jnp.ndarray):
    if a.shape == () or a.shape == (1,):
        return a.item()
    else:
        return a.tolist()


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def reduce(fn: Callable[[Carry, X], Carry], init: Carry, *xs: X) -> Carry:
    res = lax.scan(lambda carry, x: (fn(carry, *x), None), init=init, xs=xs)
    return res[0]


def flops_estimate(fn, *args):
    """Estimates the flop count of a function using XLA/HLO fanciness. See
    https://github.com/google/flax/discussions/1854"""
    m = jax.xla_computation(fn)(*args).as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    costs = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)
    return costs["flops"]


def parameter_count(model: PyTree):
    # especially with jax.vjp, we get duplicate arrays and want to uniq them
    # NB we need to use object identity here, mostly because of ShapedDtypeStruct
    leaves = {id(x): x for x in jax.tree_util.tree_leaves(model) if is_jax_array_like(x)}
    return sum(x.size for x in leaves.values())


def dump_fwd_bwd_jaxprs(out_prefix, fn, *args):
    jaxpr_vjp = jax.make_jaxpr(lambda *x: jax.vjp(fn, *x))(*args)
    primals, bkwd_fn = jax.vjp(fn, *args)
    jaxpr_bkwd_fn = jax.make_jaxpr(bkwd_fn)(primals)

    jaxpr_val_and_grad = jax.make_jaxpr(lambda *x: jax.value_and_grad(fn)(*x))(*args)

    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{out_prefix}.vg.jaxpr", "w") as f:
        f.write(jaxpr_val_and_grad.pretty_print(name_stack=True))

    with open(f"{out_prefix}.fwdbwd.jaxpr", "w") as f:
        f.write(jaxpr_vjp.pretty_print(name_stack=True))
        f.write(jaxpr_bkwd_fn.pretty_print(name_stack=True))


_orig_PRNGkey = jax.random.PRNGKey


# TODO: maybe change config option to a string value
def set_hardware_rng_ops(enabled: bool = True):
    """Enable JAX Custom PRNG extension."""
    if enabled:
        jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    else:
        jax.config.update("jax_default_prng_impl", "threefry2x32")


def global_key_array(key: PRNGKey, global_shape, mesh, mesh_axes):
    """
    Create a global array with the given key. This ensures that:
    * individual keys at positions are unique
    * the same key is made for the same position in all devices that have that position
    """

    # add key shape to global_shape and pad out axes
    global_shape = ensure_tuple(global_shape)
    orig_global_shape = global_shape
    global_shape = global_shape + key.shape
    mesh_axes = list(mesh_axes) + [None] * (len(global_shape) - len(mesh_axes))
    mesh_axes = PartitionSpec(*mesh_axes)

    assert len(global_shape) == len(mesh_axes)

    def data_callback(index: Tuple[slice, ...]):
        # we take advantage of the fact that the start indices are non-overlapping across machines (except
        # when they're identical) so we can use the index to make the keys unique
        indices = [s.indices(x) for s, x in zip(index, global_shape)]
        starts = [i[0] for i in indices]
        base_key = ft.reduce(jrandom.fold_in, (s for s in starts), key)

        assert all(i[2] == 1 for i in indices)
        lens = [i[1] - i[0] for i in indices]
        return shaped_rng_split(base_key, lens[0 : len(orig_global_shape)])

    return jax.make_array_from_callback(
        global_shape,
        jax.sharding.NamedSharding(mesh=mesh, spec=mesh_axes),
        data_callback=data_callback,
    )


@jax.tree_util.register_pytree_node_class
class pytree_partial(ft.partial):
    def tree_flatten(self):
        return ((self.func, self.args, tuple(self.keywords.values())), tuple(self.keywords.keys()))

    @classmethod
    def tree_unflatten(cls, kw_keys, tree) -> "pytree_partial":
        assert len(tree) == 3
        func, args, kw_vals = tree
        return cls(func, *args, **dict(zip(kw_keys, kw_vals)))


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


# Copy paste from equinox
def ordered_tree_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
) -> Any:
    """Like jax.tree_util.tree_map, but guaranteed to iterate over the tree
    in fixed order. (Namely depth-first left-to-right.)
    """
    # Discussion: https://github.com/patrick-kidger/equinox/issues/136
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


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
    return all(type(n) == str for n in f)


def leaf_key_paths(pytree, prefix: str = ""):
    """Creates unique, hopefully meaningful key paths for each leaf in a pytree. This is useful for
    serialization mostly. This functions knows about dicts, lists, NamedTuples, tuples, and equinox-style modules"""
    if isinstance(pytree, dict):
        return {k: leaf_key_paths(v, prefix=f"{prefix}.{k}" if prefix else k) for k, v in pytree.items()}
    elif _isnamedtupleinstance(pytree):
        d = {k: leaf_key_paths(v, prefix=f"{prefix}.{k}" if prefix else k) for k, v in pytree._asdict().items()}
        return pytree.__class__(**d)
    elif isinstance(pytree, list):
        return [leaf_key_paths(v, prefix=f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(pytree)]
    elif isinstance(pytree, tuple):
        return tuple(leaf_key_paths(v, prefix=f"{prefix}.{i}" if prefix else str(i)) for i, v in enumerate(pytree))
    elif isinstance(pytree, eqx.Module):
        values, aux = pytree.tree_flatten()
        field_names = aux[0]
        rec_values = [leaf_key_paths(v, prefix=f"{prefix}.{k}" if prefix else k) for k, v in zip(field_names, values)]

        return pytree.tree_unflatten(aux, rec_values)
    else:
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        if len(leaves) == 1:
            return jax.tree_util.tree_unflatten(treedef, [f"{prefix}"])
        else:
            return jax.tree_util.tree_unflatten(treedef, [f"{prefix}.{i}" for i in range(len(leaves))])


# from https://github.com/google/jax/issues/4285
def recursive_checkpoint(funs, threshold=2):
    if len(funs) == 1:
        return funs[0]
    elif len(funs) == 2:
        f1, f2 = funs
        return lambda x: f2(f1(x))
    elif len(funs) <= threshold:
        return functools.reduce(lambda f, g: lambda x: g(f(x)), funs)
    else:
        f1 = recursive_checkpoint(funs[: len(funs) // 2])
        f2 = recursive_checkpoint(funs[len(funs) // 2 :])
        return lambda x: f2(jax.remat(f1)(x))
