from typing import Tuple

import optax
import pytest

import jax.numpy as jnp
from jax import tree_structure, tree_flatten, tree_unflatten
from jax.experimental.maps import xmap
from psithuros.named_tensors import infer_named_axes_from_module, Array, auto_xmap
import equinox as eqx

def show_example(structured):
  flat, tree = tree_flatten(structured)
  unflattened = tree_unflatten(tree, flat)
  print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
      structured, flat, tree, unflattened))


def test_infer_named_axes():

    class MyModule(eqx.Module):
        named: Array["x", "y"]
        unnamed1: Array
        unnamed2: Array[...]
        partially_named: Array["x", ...]
        static_field: int = eqx.static_field()

    mod = MyModule(named=jnp.ones((2, 3)), unnamed1=jnp.ones((2)), unnamed2=jnp.ones((2)), partially_named=jnp.ones((2, 3)), static_field=1)
    show_example(mod)

    axes = infer_named_axes_from_module(mod)

    assert axes == [("x", "y"), (..., ), (...,), ("x", ...,)]


def test_auto_xmap_identity():
    class MyModule(eqx.Module):
        named: Array["x", "y"]
        unnamed1: Array
        unnamed2: Array[...]
        partially_named: Array["x", ...]
        static_field: int = eqx.static_field()

    mod = MyModule(named=jnp.ones((2, 3)), unnamed1=jnp.ones((2)), unnamed2=jnp.ones((2)), partially_named=jnp.ones((2, 3)), static_field=1)

    def identity(x) -> MyModule:
        return x

    fun = auto_xmap(identity)
    assert fun(mod) == mod


def test_auto_xmap_optax():
    class MyModule(eqx.Module):
        named: Array["x", "y"]
        unnamed1: Array
        unnamed2: Array[...]
        partially_named: Array["x", ...]
        static_field: int = eqx.static_field()

    mod = MyModule(named=jnp.ones((2, 3)), unnamed1=jnp.ones((2)), unnamed2=jnp.ones((2)), partially_named=jnp.ones((2, 3)), static_field=1)
    adam = optax.adam(1E-4)
    opt = adam.init(mod)

    def identity(x, opt) -> Tuple[MyModule, MyModule]:
        return x, opt

    fun = auto_xmap(identity)
    assert fun(mod, opt) == (mod, opt)

