import equinox
import jax.config
import jax.numpy as jnp
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.nn.attention import causal_mask, causal_sliding_window_attention


def test_causal_sliding_window_attention_simple():
    # test that we can't attend to something outside of the range
    D = 2
    for L, W in [(10, 5), (15, 5)]:
        SeqLen = Axis("SeqLen", L)
        Window = Axis("Window", W)
        Head = Axis("Head", D)

        keys = np.zeros((L, D), dtype=np.float32)
        keys[0, 0] = 100.0  # really want to attend to this
        values = np.zeros((L, D), dtype=np.float32)
        values[0, 1] = 300.0  # check if we did attend

        query = np.ones((L, D), dtype=np.float32)

        query = hax.named(query, (SeqLen, Head))
        keys = hax.named(keys, (SeqLen, Head))
        values = hax.named(values, (SeqLen, Head))

        result = causal_sliding_window_attention(SeqLen, Window, Head, query, keys, values)
        # we should be able to attend to the previous W positions for each position (including current), so 6-10 can't attend
        # to 0-4 and can't get the 100.0 key
        result = result.rearrange((SeqLen, Head)).array
        assert jnp.allclose(result[0:W, 1], 300)
        assert jnp.allclose(result[W:, 1], 0)


def test_sliding_window_attention_fancier():
    # jax.config.update("jax_disable_jit", True)
    D = 4
    for L, W in [(2, 1), (2, 2), (4, 2), (10, 5), (15, 5), (16, 2), (15, 3), (10, 10)]:
        SeqLen = Axis("SeqLen", L)
        Window = Axis("Window", W)
        Head = Axis("Head", D)

        q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)

        query = hax.random.uniform(q_key, (SeqLen, Head))
        keys = hax.random.uniform(k_key, (SeqLen, Head))
        values = hax.random.uniform(v_key, (SeqLen, Head))

        result = causal_sliding_window_attention(SeqLen, Window, Head, query, keys, values)
        result = result.rearrange((SeqLen, Head)).array

        KSeqLen = Axis("KSeqLen", SeqLen.size)
        keys = keys.rename({SeqLen: KSeqLen})
        values = values.rename({SeqLen: KSeqLen})

        diff = hax.arange(SeqLen).broadcast_axis(KSeqLen) - hax.arange(KSeqLen).broadcast_axis(SeqLen)
        mask = causal_mask(SeqLen, KSeqLen) & (diff < Window.size) & (diff >= 0)

        # check that the result is the same as non-blocked attention with the right mask
        expected = hax.nn.attention.dot_product_attention(SeqLen, KSeqLen, Head, query, keys, values, mask=mask)

        expected = expected.rearrange((SeqLen, Head)).array

        assert jnp.allclose(result, expected)


def test_longformer_alibi_bias_pos_invariance():
    D = 1
    W = 32
    H = 1

    L = 4096

    Head = Axis("Head", H)
    SeqLen = Axis("SeqLen", L)
    Window = Axis("Window", W)
    Hidden = Axis("Hidden", D)

    # this cycles [31, ..., 0, 31, ..., 0, ...]
    cycle = np.flip(np.arange(W, dtype=np.float32))
    v = np.tile(cycle, L // W).reshape((L, H, D))
    v = hax.named(v, (SeqLen, Head, Hidden))

    q = hax.ones((SeqLen, Head, Hidden), dtype=jnp.bfloat16) * 0.001
    k = hax.ones((SeqLen, Head, Hidden), dtype=jnp.bfloat16) * 0.001

    # bias gets geometrically larger as we go further in the sequence
    # this is especially true if there are a lot of heads
    big_head = hax.Axis("Head", 16)
    # NB: this test doesn't work if you use bfloat16 for biases
    bias = hax.nn.attention.alibi_bias(big_head, SeqLen, dtype=jnp.float32).slice(big_head, Head, 0)

    attn = causal_sliding_window_attention(SeqLen, Window, Hidden, q, k, v, bias=bias, attention_dtype=jnp.bfloat16)
    attn = attn.rearrange((SeqLen, Head, Hidden)).array.reshape(L)

    # final value for each cycle should be the same
    finals = attn[W - 1 :: W]
    assert np.isclose(finals, finals[0], rtol=2e-4).all(), f"finals: {finals}"


def test_longformer_grad():
    D = 4
    W = 32
    H = 2

    L = 128

    Head = Axis("Head", H)
    SeqLen = Axis("SeqLen", L)
    Window = Axis("Window", W)
    Hidden = Axis("Hidden", D)

    def grad_sliding_window_attention(q, k, v):
        return equinox.filter_grad(
            lambda x: causal_sliding_window_attention(SeqLen, Window, Hidden, x, k, v).sum().scalar()
        )(q)

    q = hax.random.uniform(jax.random.PRNGKey(0), (SeqLen, Head, Hidden))
    k = hax.random.uniform(jax.random.PRNGKey(1), (SeqLen, Head, Hidden))
    v = hax.random.uniform(jax.random.PRNGKey(2), (SeqLen, Head, Hidden))

    grad_q = grad_sliding_window_attention(q, k, v)  # noqa: F841
