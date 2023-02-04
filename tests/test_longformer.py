import jax.config
import jax.numpy as jnp
import numpy as np

import haliax as hax
from haliax import Axis
from levanter.models.longformer import _ignore_padding_attn_mask, causal_sliding_window_attention2


def test_ignore_padding_mask():

    for (sz, w) in [(10, 5), (10, 4), (10, 10), (10, 9), (10, 1)]:
        SeqLen = Axis("SeqLen", sz)
        Window = Axis("Window", w)

        mask = _ignore_padding_attn_mask(SeqLen, Window).rearrange((SeqLen, Window)).array

        assert mask[0, Window.size - 1]
        assert jnp.all(hax.logical_not(mask[0, : Window.size - 1]))
        assert jnp.all(mask[Window.size :, :])
        assert jnp.all(mask[Window.size - 1, :])

        assert jnp.sum(hax.logical_not(mask).astype(jnp.int32)) == sum(range(Window.size))


def test_causal_sliding_window_attention():
    # test that we can't attend to something outside of the range
    L, W, D = 15, 5, 2
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

    jax.config.update("jax_disable_jit", True)
    result = causal_sliding_window_attention2(SeqLen, Window, Head, query, keys, values)
    # we should be able to attend to the previous W positions for each position (including current), so 6-10 can't attend
    # to 0-4 and can't get the 100.0 key
    result = result.rearrange((SeqLen, Head)).array
    assert jnp.allclose(result[0:W, 1], 300)
    assert jnp.allclose(result[W:, 1], 0)
