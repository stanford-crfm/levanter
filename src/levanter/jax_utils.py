from pathlib import Path
from typing import Callable, Optional, Sequence, TypeVar, Union

import equinox as eqx
import jax
import numpy as np
from chex import PRNGKey
from equinox.custom_types import PyTree
from jax import lax
from jax import numpy as jnp
from jax import random as jrandom


def maybe_rng_split(key: Optional[PRNGKey], num: int = 2):
    """Splits a random key into multiple random keys. If the key is None, then it replicates the None. Also handles
    num == 1 case"""
    if key is None:
        return [None] * num
    elif num == 1:
        return jnp.reshape(key, (1,) + key.shape)
    else:
        return jrandom.split(key, num)


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


def jnp_to_python(a: jnp.ndarray):
    if a.shape == () or a.shape == (1,):
        return a.item()
    else:
        return a.tolist()


Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


def fold_left(fn: Callable[[Carry, X], Carry], init: Carry, *xs: X) -> Carry:
    res = lax.scan(lambda carry, x: (fn(carry, *x), None), init=init, xs=xs)
    return res[0]


def flops_estimate(fn, *args):
    """Estimates the flop count of a function using XLA/HLO fanciness. See https://github.com/google/flax/discussions/1854"""
    m = jax.xla_computation(fn)(*args).as_hlo_module()
    client = jax.lib.xla_bridge.get_backend()
    costs = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)
    return costs["flops"]


def backward_graph_size(fn, *args):
    """
    Estimates the size of the backward graph of a function, in terms of number of parameters.
    This will sometimes overestimate the size of the graph, for two (known) reasons:
      1. the provenance of constants hiding inside jitted functions are hard to track
      2. it's possible that XLA will further optimize some of the code beyond what I can see.

    vjp (which is the "forward" pass) returns a pytree fn that contains everything needed to compute the backward
    pass, but it includes the parameters and inputs that are needed in the backward pass. But we already "pay"
    for those in the forward pass/the parameter count, so we don't need to count them twice.
    """

    # first fine parameters/inputs that we've already priced in (parameters, inputs)
    input_leaves = jax.tree_leaves((fn, args))
    input_leaf_ids = {id(x): x for x in input_leaves}

    faxpr = jax.make_jaxpr(fn)(*args)
    # fold in consts that are only in the jaxpr (and not part of the pytree)
    for const in faxpr.consts:
        input_leaf_ids[id(const)] = const

    dynamic, static = eqx.partition((fn, args), eqx.is_array_like)

    def part_fn(dynamic):
        fn, args = eqx.combine(dynamic, static)
        return fn(*args)

    primals, bkwd_fn = jax.vjp(part_fn, dynamic)

    vjp_leaves = jax.tree_leaves(bkwd_fn)
    new_leaves = {id(x): x for x in vjp_leaves if id(x) not in input_leaf_ids}

    return parameter_count(list(new_leaves.values()))


def dump_jaxpr(file, fn, *args, **kwargs):
    jaxpr = jax.make_jaxpr(fn)(*args, **kwargs)
    with open(file, "w") as f:
        f.write(jaxpr.pretty_print(source_info=True, name_stack=True))


def parameter_count(model: PyTree):
    def _is_param_leaf(x):
        return (isinstance(x, jax.ShapeDtypeStruct) and jnp.issubdtype(x.dtype, jnp.inexact)) or eqx.is_inexact_array(
            x
        )

    # especially with jax.vjp, we get duplicate arrays and want to uniq them
    # NB we need to use object identity here, mostly because of ShapedDtypeStruct
    leaves = {id(x): x for x in jax.tree_leaves(model) if _is_param_leaf(x)}
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


def get_nth_rank(pytree, rank=0, leaf_filter=eqx.is_inexact_array):
    return jax.tree_map(lambda leaf: leaf[rank] if leaf_filter(leaf) else leaf, pytree)
