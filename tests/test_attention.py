import jax.numpy as jnp

import haliax as hax

from levanter.models.attention import CausalAttentionMask


def test_causal_mask_blocking():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = CausalAttentionMask(pos, key_pos)

    blocked_mask = mask.blocked(pos, 16).blocked(key_pos, 16)
    assert blocked_mask.Pos.size == 128 // 16
    assert blocked_mask.KeyPos.size == 128 // 16

    mat_blocked = blocked_mask.materialize()

    assert hax.all(mat_blocked == hax.nn.attention.causal_mask(pos.resize(8), key_pos.resize(8)))

    mat_mask = mask.materialize()

    for i in range(8):
        for j in range(8):
            assert mat_blocked.array[i, j] == jnp.any(mat_mask.array[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16])


def test_causal_mask_slicing():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = CausalAttentionMask(pos, key_pos)

    sliced = mask.slice(pos, 7, length=16).slice(key_pos, 24, length=16)

    mat_mask = mask.materialize()
    mat_sliced = sliced.materialize()

    for i in range(16):
        for j in range(16):
            assert mat_sliced.array[i, j] == mat_mask.array[7 + i, 24 + j]
