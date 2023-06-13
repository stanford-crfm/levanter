import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
import haliax.tree_util as htu
from haliax import Axis


def test_resize_axis():

    A = hax.Axis("A", 10)
    B = hax.Axis("B", 20)
    C = hax.Axis("C", 30)

    class Module(eqx.Module):
        name1: hax.NamedArray
        name2: hax.NamedArray
        name3: hax.NamedArray

    module = Module(
        name1=hax.random.normal(jax.random.PRNGKey(0), (B, A, C)),
        name2=hax.zeros((B, C)),
        name3=hax.zeros((Axis("A", 20),)),
    )

    NewA = A.resize(15)

    module2 = htu.resize_axis(module, NewA, jax.random.PRNGKey(1))

    assert module2.name1.axes == (B, NewA, C)
    assert module2.name2.axes == (B, C)
    assert module2.name3.axes == (NewA,)

    # we don't mess with the mean or std of the original array too much
    assert jnp.allclose(module2.name1.mean().array, module.name1.mean().array, rtol=1e-1, atol=1e-2)
