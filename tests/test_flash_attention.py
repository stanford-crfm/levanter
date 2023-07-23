import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn

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

    mask = hax.nn.attention.causal_mask(QPos, KPos)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Key))
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Key))
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Key))

    flash_out = flash_attention(QPos, KPos, Key, q, k, v, inference=True, mask=mask)
    hax_out = hnn.attention.dot_product_attention(QPos, KPos, Key, q, k, v, mask=mask)

    assert hax_out.axes == flash_out.axes
    assert jnp.allclose(hax_out.array, flash_out.array, atol=1e-5, rtol=1e-5)
