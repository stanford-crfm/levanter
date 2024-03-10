import jax.numpy as jnp
import pytest

import haliax as hax

from levanter.models.attention import AttentionMask, _te_bin_and_group_axes_by_function, _te_flash_attention
from test_utils import skip_if_module_missing


@pytest.mark.skip
def test_causal_mask_blocking():
    pos = hax.Axis("pos", 128)
    key_pos = pos.alias("key_pos")

    mask = AttentionMask.causal()

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

    mask = AttentionMask.causal()

    mat_mask = mask.materialize(pos, key_pos)
    mat_sliced = mask.materialize(pos, key_pos, q_slice=hax.dslice(7, 16), k_slice=hax.dslice(24, 16))

    for i in range(16):
        for j in range(16):
            assert mat_sliced.array[i, j] == mat_mask.array[7 + i, 24 + j]


def test_te_bin_and_group_axes_by_function():
    QPos = hax.Axis("QPos", 128)
    KPos = hax.Axis("KPos", 128)
    D = hax.Axis("D", 64)
    H = hax.Axis("H", 8)
    B = hax.Axis("B", 32)
    G = hax.Axis("G", 4)

    q = hax.zeros((B, QPos, H, D))
    k = hax.zeros((B, KPos, H, D))
    v = hax.zeros((B, KPos, H, D))

    q_c, k_c, v_c = _te_bin_and_group_axes_by_function(q, k, v, "QPos", "KPos", "D")
    assert q_c["B"] == [B]
    assert k_c["B"] == [B]
    assert v_c["B"] == [B]

    assert q_c["S"] == [QPos]
    assert k_c["S"] == [KPos]
    assert v_c["S"] == [KPos]

    assert q_c["H"] == [H]
    assert k_c["H"] == [H]
    assert v_c["H"] == [H]

    assert q_c["D"] == [D]
    assert k_c["D"] == [D]
    assert v_c["D"] == [D]

    gq = hax.zeros((B, QPos, H, G, D))
    q_c, k_c, v_c = _te_bin_and_group_axes_by_function(gq, k, v, "QPos", "KPos", "D")
    assert q_c["H"] == [H, G]
    assert k_c["H"] == [H]
    assert v_c["H"] == [H]

    gk = hax.zeros((B, KPos, G, H, D))
    with pytest.raises(ValueError):
        _te_bin_and_group_axes_by_function(q, gk, v, "QPos", "KPos", "D")

    with pytest.raises(ValueError):
        _te_bin_and_group_axes_by_function(gq, gk, v, "QPos", "KPos", "D")

    for gk_axes in [(B, KPos, G, H, D), (B, KPos, G, H, D), (G, B, KPos, H, D)]:
        gk = hax.zeros(gk_axes)
        q_c, k_c, v_c = _te_bin_and_group_axes_by_function(gq, gk, gk, "QPos", "KPos", "D")
        assert q_c["H"] == [H, G]
        assert k_c["H"] == [H, G]
        assert v_c["H"] == [H, G]

    # axes that come before QPos are treated as batch (if shared)
    gq = hax.zeros((G, B, QPos, H, D))
    for gk_axes in [(B, KPos, H, G, D), (B, KPos, G, H, D), (G, B, KPos, H, D)]:
        gk = hax.zeros(gk_axes)
        q_c, k_c, v_c = _te_bin_and_group_axes_by_function(gq, gk, gk, "QPos", "KPos", "D")
        assert q_c["H"] == [H]
        assert k_c["H"] == [H]
        assert v_c["H"] == [H]
        assert q_c["B"] == [G, B]
        assert k_c["B"] == [G, B]
        assert v_c["B"] == [G, B]


def test_mqa_te_bin_and_group_axes_by_function():
    B = hax.Axis("B", 32)
    QPos = hax.Axis("QPos", 128)
    KPos = hax.Axis("KPos", 128)
    D = hax.Axis("D", 64)
    G = hax.Axis("G", 4)

    # MQA
    gq = hax.zeros((B, QPos, G, D))
    k = hax.zeros((B, KPos, D))
    v = hax.zeros((B, KPos, D))

    q_c, k_c, v_c = _te_bin_and_group_axes_by_function(gq, k, v, "QPos", "KPos", "D")
    assert q_c["H"] == [G]
    assert k_c["H"] == []
    assert v_c["H"] == []

    gk = hax.zeros((B, KPos, G, D))
    with pytest.raises(ValueError):
        # don't currently handle dim in Q and K but not V
        _te_bin_and_group_axes_by_function(gq, gk, v, "QPos", "KPos", "D")


@skip_if_module_missing("transformer_engine")
@pytest.mark.parametrize("q_heads", [1, 2, 4])
def test_llama_attention_uses_te(q_heads):
    QPos = hax.Axis("position", 128)
    KPos = hax.Axis("key_position", 128)
    B = hax.Axis("batch", 8)
    Head = hax.Axis("kv_heads", 8)
    D = hax.Axis("head_size", 64)
    Q_Head = hax.Axis("q_heads_per_group", q_heads)
    mask = AttentionMask.causal()

    q = hax.zeros((B, Head, Q_Head, QPos, D), dtype=jnp.bfloat16)
    k = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)
    v = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)

    # mostly testing that this doesn't crash
    out = _te_flash_attention(
        "position",
        "key_position",
        "head_size",
        q,
        k,
        v,
        mask,
        attention_dtype=jnp.bfloat16,
    )

    assert jnp.allclose(out, 0.0)


@skip_if_module_missing("transformer_engine")
def test_gpt2_attention_uses_te():
    QPos = hax.Axis("position", 128)
    KPos = hax.Axis("key_position", 128)
    B = hax.Axis("batch", 8)
    Head = hax.Axis("heads", 8)
    D = hax.Axis("head_size", 64)
    mask = AttentionMask.causal()

    q = hax.zeros((B, Head, QPos, D), dtype=jnp.bfloat16)
    k = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)
    v = hax.zeros((B, Head, KPos, D), dtype=jnp.bfloat16)

    out = _te_flash_attention(
        "position",
        "key_position",
        "head_size",
        q,
        k,
        v,
        mask,
        attention_dtype=jnp.bfloat16,
    )
    assert jnp.allclose(out, 0.0)
