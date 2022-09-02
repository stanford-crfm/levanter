from functools import wraps

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import NamedArray


def _compare_eqx_and_haliax(hax_mod: eqx.Module, eqx_mod: eqx.Module):
    @wraps(hax_mod.__call__)
    def f(x: NamedArray, *args, **kwargs):
        unnamed_x = x.array
        hax_out = hax_mod(x, *args, **kwargs)  # type: ignore
        eqx_out = eqx_mod(unnamed_x, *args, **kwargs)  # type: ignore

        assert jnp.allclose(hax_out.array, eqx_out)
        return hax_out

    return f


def test_layer_norm():
    H = hax.Axis("H", 10)
    hax_ln = hax.nn.LayerNorm(H)
    eqx_ln = eqx.nn.LayerNorm(shape=(H.size,))

    f = _compare_eqx_and_haliax(hax_ln, eqx_ln)
    out = f(hax.random.uniform(jrandom.PRNGKey(0), (H,)))

    assert out.axes == (H,)


def test_dropout():
    H = hax.Axis("H", 10)
    key = jrandom.PRNGKey(0)
    hax_dropout = hax.nn.Dropout(0.5)
    eqx_dropout = eqx.nn.Dropout(0.5)

    f = _compare_eqx_and_haliax(hax_dropout, eqx_dropout)
    out = f(hax.random.uniform(jrandom.PRNGKey(0), (H,)), key=key, inference=False)

    assert out.axes == (H,)
