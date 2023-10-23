import functools

import equinox
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn

from levanter.models.attention import AttentionMask
from levanter.models.flash_attention import BLOCK_SIZE, flash_attention


def test_flash_attention_acausal():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    flash_out = flash_attention(QPos, KPos, Key, q, k, v, inference=True)
    hax_out = hnn.attention.dot_product_attention(QPos, KPos, Key, q, k, v)

    assert hax_out.axes == flash_out.axes
    assert jnp.allclose(hax_out.array, flash_out.array, atol=1e-5, rtol=1e-5)


def test_flash_attention_causal_mask():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    mask = AttentionMask.causal()

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    flash_out = flash_attention(QPos, KPos, Key, q, k, v, inference=True, mask=mask)
    hax_out = hnn.attention.dot_product_attention(QPos, KPos, Key, q, k, v, mask=mask.materialize(QPos, KPos))

    assert hax_out.axes == flash_out.axes
    assert jnp.allclose(hax_out.array, flash_out.array, atol=1e-5, rtol=1e-5)


def test_grad_attention():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    mask = hax.nn.attention.causal_mask(QPos, KPos)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    @equinox.filter_value_and_grad
    def d_attn(qkv, fn):
        q, k, v = qkv
        x_out = fn(QPos, KPos, Key, q, k, v, mask=mask)
        return (x_out * x_out).sum().scalar()

    hax_val, (hax_dq, hax_dk, hax_dv) = d_attn((q, k, v), hnn.attention.dot_product_attention)
    fa_val, (fa_dq, fa_dk, fa_dv) = d_attn((q, k, v), functools.partial(flash_attention, inference=True))

    assert jnp.allclose(hax_val, fa_val, atol=1e-4, rtol=1e-4)
    assert hax_dq.axes == fa_dq.axes
    assert hax_dk.axes == fa_dk.axes
    assert hax_dv.axes == fa_dv.axes

    assert jnp.allclose(hax_dq.array, fa_dq.array, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(hax_dk.array, fa_dk.array, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(hax_dv.array, fa_dv.array, atol=1e-4, rtol=1e-4)


def test_fa_dropout_does_something():
    Key = hax.Axis("Key", 8)
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    mask = hax.nn.attention.causal_mask(QPos, KPos)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    p_drop = 0.5

    fa_with_dropout = functools.partial(flash_attention, inference=False, dropout=p_drop, key=jrandom.PRNGKey(3))
    fa_without_dropout = functools.partial(flash_attention, inference=True)

    without_o = fa_without_dropout(QPos, KPos, Key, q, k, v, mask=mask)
    with_o = fa_with_dropout(QPos, KPos, Key, q, k, v, mask=mask)

    assert with_o.axes == without_o.axes
    assert not jnp.any(jnp.isclose(with_o.array, without_o.array, atol=1e-5, rtol=1e-5))
