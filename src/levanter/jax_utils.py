import functools as ft
from pathlib import Path
from typing import Callable, Tuple, TypeVar

import equinox as eqx
import jax
from chex import PRNGKey
from jax import lax
from jax import numpy as jnp
from jax import random as jrandom
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
