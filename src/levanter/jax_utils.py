import functools as ft
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar

import equinox as eqx
import jax
from chex import PRNGKey
from jax import lax
from jax import numpy as jnp
from jax import random as jrandom
from jax.experimental import multihost_utils
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.interpreters.pxla import PartitionSpec
from jaxtyping import PyTree

from haliax.jax_utils import shaped_rng_split


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
    def _is_param_leaf(x):
        return (isinstance(x, jax.ShapeDtypeStruct) and jnp.issubdtype(x.dtype, jnp.inexact)) or eqx.is_inexact_array(
            x
        )

    # especially with jax.vjp, we get duplicate arrays and want to uniq them
    # NB we need to use object identity here, mostly because of ShapedDtypeStruct
    leaves = {id(x): x for x in jax.tree_util.tree_leaves(model) if _is_param_leaf(x)}
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


def global_key_array(key: PRNGKey, global_shape, global_mesh, mesh_axes):
    """
    Create a global array with the given key. This ensures that:
    * individual keys at positions are unique
    * the same key is made for the same position in all devices that have that position
    """

    # add key shape to global_shape and pad out axes
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

    return GlobalDeviceArray.from_callback(
        global_shape=global_shape,
        global_mesh=global_mesh,
        mesh_axes=mesh_axes,
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


def multihost_broadcast_obj(obj, is_source: Optional[bool] = None):
    """Very hacky way to broadcast an arbitrary pickleable object from the given host to all hosts.
    This is useful for broadcasting things like wandb run id names. This is very slow, but it's only
    used for initialization so it's not a big deal."""
    # TODO: see if we can access underlying jax communication mechanism to do this more efficiently
    import pickle

    if is_source is None:
        is_source = jax.process_index() == 0

    BUF_SIZE = 16 * 1024
    HEADER_SIZE = 4  # 4 bytes for length of packet. overkill, but whatever
    CONTENT_SIZE = BUF_SIZE - HEADER_SIZE

    buf = jnp.zeros(BUF_SIZE, dtype=jnp.uint8)
    if is_source:
        pickled = pickle.dumps(obj)
        # need to be careful here. we have to always send at least one packet, even if it's empty
        # so that the other side knows we're done
        if len(pickled) == 0:
            multihost_utils.broadcast_one_to_all(buf, is_source)
        else:
            if len(pickled) == 0:
                content_length = 0
                header = content_length.to_bytes(HEADER_SIZE, "big")
                pad_length = CONTENT_SIZE
                buf = jnp.array(header + b"\0" * pad_length, dtype=jnp.uint8)
                multihost_utils.broadcast_one_to_all(buf, is_source=is_source)
            else:
                # send BUF_SIZE at a time including header
                for i in range(0, len(pickled), CONTENT_SIZE):
                    content_length = min(CONTENT_SIZE, len(pickled) - i)
                    pad_length = CONTENT_SIZE - content_length
                    header = content_length.to_bytes(HEADER_SIZE, "big")
                    # have to convert to list because numpy will attempt to parse as ascii
                    buf = jnp.array(
                        list(header + pickled[i : i + content_length] + b"\0" * pad_length), dtype=jnp.uint8
                    )
                    multihost_utils.broadcast_one_to_all(buf, is_source=is_source)
    else:
        pickled = b""
        # build up the pickled string
        while True:
            buf = multihost_utils.broadcast_one_to_all(buf, is_source=is_source)
            content_length = int.from_bytes(buf[:HEADER_SIZE], "big")
            pickled += bytes(buf[HEADER_SIZE : HEADER_SIZE + content_length])
            if content_length < CONTENT_SIZE:
                break

        # depickle:
        obj = pickle.loads(pickled)

    return obj


def simplify_gdas(pytree: PyTree):
    """Simplify fully-replicated global device arrays to simple arrays. Typically this is for scalars or small arrays
    that we want to log"""

    def _simplify_gda(gda):
        if isinstance(gda, GlobalDeviceArray):
            if gda.is_fully_replicated:
                return gda.local_data(0)
            return gda
        else:
            return gda

    return jax.tree_map(_simplify_gda, pytree)


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
