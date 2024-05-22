import jax
import jax.numpy as jnp
import numpy as np
from chex import assert_trees_all_close

import haliax as hax
from haliax import Axis
from haliax.nn.attention import causal_mask

from levanter.models.longformer import causal_sliding_window_attention


def test_causal_sliding_window_attention_simple():
    # test that we can't attend to something outside of the range
    D = 2
    for L, W in [(10, 5), (15, 5)]:
        Pos = Axis("Pos", L)
        Window = Axis("Window", W)
        Head = Axis("Head", D)

        keys = np.zeros((L, D), dtype=np.float32)
        keys[0, 0] = 100.0  # really want to attend to this
        values = np.zeros((L, D), dtype=np.float32)
        values[0, 1] = 300.0  # check if we did attend

        query = np.ones((L, D), dtype=np.float32)

        query = hax.named(query, (Pos, Head))
        keys = hax.named(keys, (Pos, Head))
        values = hax.named(values, (Pos, Head))

        result = causal_sliding_window_attention(Pos, Window, Head, query, keys, values)
        # we should be able to attend to the previous W positions for each position (including current), so 6-10 can't attend
        # to 0-4 and can't get the 100.0 key
        result = result.rearrange((Pos, Head)).array
        assert_trees_all_close(result[0:W, 1], 300)
        assert_trees_all_close(result[W:, 1], 0)


def test_sliding_window_attention_fancier():
    D = 4
    for L, W in [(2, 1), (2, 2), (4, 2), (10, 5), (15, 5), (16, 2), (15, 3), (10, 10)]:
        Pos = Axis("Pos", L)
        Window = Axis("Window", W)
        Head = Axis("Head", D)

        q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(0), 3)

        query = hax.random.uniform(q_key, (Pos, Head))
        keys = hax.random.uniform(k_key, (Pos, Head))
        values = hax.random.uniform(v_key, (Pos, Head))

        result = causal_sliding_window_attention(Pos, Window, Head, query, keys, values)
        result = result.rearrange((Pos, Head)).array

        KPos = Axis("KPos", Pos.size)
        keys = keys.rename({Pos: KPos})
        values = values.rename({Pos: KPos})

        diff = hax.arange(Pos).broadcast_axis(KPos) - hax.arange(KPos).broadcast_axis(Pos)
        mask = causal_mask(Pos, KPos) & (diff < Window.size) & (diff >= 0)

        # check that the result is the same as non-blocked attention with the right mask
        expected = hax.nn.attention.dot_product_attention(KPos, Head, query, keys, values, mask=mask)

        expected = expected.rearrange((Pos, Head)).array

        assert_trees_all_close(result, expected, atol=1e-3, rtol=1e-3)


def test_longformer_alibi_bias_pos_invariance():
    D = 1
    W = 32
    H = 1

    L = 4096

    Head = Axis("Head", H)
    Pos = Axis("Pos", L)
    Window = Axis("Window", W)
    Hidden = Axis("Hidden", D)

    # this cycles [31, ..., 0, 31, ..., 0, ...]
    cycle = np.flip(np.arange(W, dtype=np.float32))
    v = np.tile(cycle, L // W).reshape((L, H, D))
    v = hax.named(v, (Pos, Head, Hidden))

    q = hax.ones((Pos, Head, Hidden), dtype=jnp.bfloat16) * 0.001
    k = hax.ones((Pos, Head, Hidden), dtype=jnp.bfloat16) * 0.001

    # bias gets geometrically larger as we go further in the sequence
    # this is especially true if there are a lot of heads
    big_head = hax.Axis("Head", 16)
    # NB: this test doesn't work if you use bfloat16 for biases
    bias = hax.nn.attention.alibi_attention_bias(big_head, Pos, dtype=jnp.float32).slice(big_head, Head, 0)

    attn = causal_sliding_window_attention(Pos, Window, Hidden, q, k, v, bias=bias, attention_dtype=jnp.bfloat16)
    attn = attn.rearrange((Pos, Head, Hidden)).array.reshape(L)

    # final value for each cycle should be the same
    finals = attn[W - 1 :: W]
    assert np.isclose(finals, finals[0], rtol=2e-4).all(), f"finals: {finals}"
