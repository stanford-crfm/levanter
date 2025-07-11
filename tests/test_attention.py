import math

import equinox
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from chex import assert_trees_all_close
from jax.sharding import Mesh

import haliax as hax
from haliax import Axis, NamedArray

from levanter.layers.attention import (
    Attention,
    AttentionBackend,
    AttentionConfig,
    AttentionMask,
    KvPageCache,
    _bin_and_group_axes_by_function,
    _te_flash_attention,
    _tpu_splash_attention,
    dot_product_attention,
)
from levanter.layers.page_table import PageBatchInfo, PageTable
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

    q_c, k_c, v_c = _bin_and_group_axes_by_function(q, k, v, "QPos", "KPos", "D")
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
    q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, k, v, "QPos", "KPos", "D")
    assert q_c["H"] == [H, G]
    assert k_c["H"] == [H]
    assert v_c["H"] == [H]

    gk = hax.zeros((B, KPos, G, H, D))
    with pytest.raises(ValueError):
        _bin_and_group_axes_by_function(q, gk, v, "QPos", "KPos", "D")

    with pytest.raises(ValueError):
        _bin_and_group_axes_by_function(gq, gk, v, "QPos", "KPos", "D")

    for gk_axes in [(B, KPos, G, H, D), (B, KPos, G, H, D), (G, B, KPos, H, D)]:
        gk = hax.zeros(gk_axes)
        q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, gk, gk, "QPos", "KPos", "D")
        assert q_c["H"] == [H, G]
        assert k_c["H"] == [H, G]
        assert v_c["H"] == [H, G]

    # axes that come before QPos are treated as batch (if shared)
    gq = hax.zeros((G, B, QPos, H, D))
    for gk_axes in [(B, KPos, H, G, D), (B, KPos, G, H, D), (G, B, KPos, H, D)]:
        gk = hax.zeros(gk_axes)
        q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, gk, gk, "QPos", "KPos", "D")
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

    q_c, k_c, v_c = _bin_and_group_axes_by_function(gq, k, v, "QPos", "KPos", "D")
    assert q_c["H"] == [G]
    assert k_c["H"] == []
    assert v_c["H"] == []

    gk = hax.zeros((B, KPos, G, D))
    with pytest.raises(ValueError):
        # don't currently handle dim in Q and K but not V
        _bin_and_group_axes_by_function(gq, gk, v, "QPos", "KPos", "D")


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
        scaling_factor=1 / math.sqrt(D.size),
    )

    assert_trees_all_close(out.array, 0.0)


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
        scaling_factor=1 / math.sqrt(D.size),
    )
    assert_trees_all_close(out.array, 0.0)


def test_tpu_splash_attention():
    if jax.default_backend() != "tpu":
        pytest.skip("TPU only")

    BLOCK_SIZE = 512

    Head = hax.Axis("Head", 8)
    Key = hax.Axis("Key", 128)  # splash only supports 128
    QPos = hax.Axis("QPos", BLOCK_SIZE * 2)
    KPos = hax.Axis("KPos", BLOCK_SIZE * 2)

    q = hax.random.normal(jrandom.PRNGKey(0), (QPos, Head, Key)) * 0.02
    k = hax.random.normal(jrandom.PRNGKey(1), (KPos, Head, Key)) * 0.02
    v = hax.random.normal(jrandom.PRNGKey(2), (KPos, Head, Key)) * 0.02

    mask = AttentionMask.causal()

    with jax.sharding.Mesh(jax.devices(), ("dp",)):
        flash_out = _tpu_splash_attention(
            QPos,
            KPos,
            Key,
            q,
            k,
            v,
            inference=True,
            mask=mask,
            block_size=BLOCK_SIZE,
            scaling_factor=1 / math.sqrt(Head.size),
        )
        hax_out = hax.nn.attention.dot_product_attention(KPos, Key, q, k, v, mask=mask.materialize(QPos, KPos))
        assert hax_out.axes == flash_out.axes
        assert_trees_all_close(hax_out.array, flash_out.array, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("impl", ["default", "jax_flash", "vanilla"])
def test_segment_ids_are_respected(impl):
    # test that we can't attend to something outside of the range
    # splash needs 128
    D = 128 if impl == "default" else 2
    L = 256
    Pos = Axis("Pos", L)
    Head = Axis("Head", D)

    keys = np.zeros((L, D), dtype=np.float32)
    keys[0, 0] = 100.0  # really want to attend to this
    values = np.zeros((L, D), dtype=np.float32)
    values[0, 1] = 300.0  # check if we did attend
    KPos = Pos.alias("KPos")

    query = np.ones((L, D), dtype=np.float32)

    query = hax.named(query, (Pos, Head))
    keys = hax.named(keys, (KPos, Head))
    values = hax.named(values, (KPos, Head))

    segment_ids = np.array([0, 0, 0] + [1] * (L - 3), dtype=np.int32)
    segment_ids = jax.device_put(segment_ids, jax.sharding.PositionalSharding(jax.devices()))
    segment_ids = hax.named(segment_ids, (Pos,))
    mask = AttentionMask(causal_offset=0, segment_ids=segment_ids)

    devices = jax.devices()

    with Mesh(devices, ("dp",)):
        result = jit_dpa(
            Pos, KPos, Head, query, keys, values, attn_backend=AttentionBackend(impl), mask=mask, flash_block_size=128
        )

    # the first 3 positions should all have a value of 300.0
    assert_trees_all_close(result.array[0:3, 1], 300.0, atol=1e-3, rtol=1e-3)
    # the rest should be 0
    assert_trees_all_close(result.array[3:, 1], 0.0, atol=1e-3, rtol=1e-3)


# TODO: fix flash attention for offsets
@pytest.mark.parametrize("impl", ["vanilla"])
def test_causal_offset_cross_attention(impl):
    """Verify that a positive causal *offset* relaxes the masking during cross-attention.

    We compare the output of ``dot_product_attention`` when provided the structured
    ``AttentionMask`` with *offset* against the output obtained when passing the
    *materialised* boolean mask explicitly – they should be identical.
    """

    offset = 2
    FullPos = Axis("pos", 6)
    Pos = FullPos.resize(offset)
    KeyPos = Axis("key_pos", 6)
    Head = Axis("head", 2)
    KeyDim = Axis("embed", 4)

    k = hax.random.normal(jrandom.PRNGKey(1), (KeyPos, Head, KeyDim))
    v = hax.random.normal(jrandom.PRNGKey(2), (KeyPos, Head, KeyDim))
    q = hax.random.normal(jrandom.PRNGKey(0), (FullPos, Head, KeyDim))
    q_sub = q["pos", 4:6]

    struct_mask = AttentionMask.causal(offset=FullPos.size - offset)

    offset_out = jit_dpa(
        Pos,
        KeyPos,
        KeyDim,
        q_sub,
        k,
        v,
        mask=struct_mask,
        inference=True,
        attn_backend=AttentionBackend(impl),
        flash_block_size=1,
    )

    mask = AttentionMask.causal()

    full_out = jit_dpa(
        FullPos,
        KeyPos,
        KeyDim,
        q,
        k,
        v,
        mask=mask,
        flash_block_size=1,
    )

    # The output should be the same, since the mask is relaxed by the offset
    assert_trees_all_close(offset_out.array, full_out.array[4:6, :], atol=1e-4, rtol=1e-4)

    # sanity check: output should be wrong if we don't use the offset
    wrong_out = jit_dpa(
        Pos,
        KeyPos,
        KeyDim,
        q_sub,
        k,
        v,
        mask=mask,
        inference=True,
        attn_backend=AttentionBackend(impl),
        flash_block_size=1,
    )

    assert not jnp.allclose(
        offset_out.array, wrong_out.array, atol=1e-4, rtol=1e-4
    ), "Output should differ without offset"


def test_attention_decode_matches_full_ar():
    """Ensure incremental decode matches full-sequence forward pass."""
    B = Axis("batch", 1)
    Pos = Axis("position", 4)
    Embed = Axis("embed", 8)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2

    # Full forward pass
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    # Incremental decode ---------------------------------------------------
    cache = cfg.empty_kv_cache(B, Pos, dtype=jnp.float32)
    out_chunks = []
    for i in range(Pos.size):
        x_tok = x[Pos, hax.dslice(i, 1)]
        sub_pos = x_tok.resolve_axis("position")
        pos_ids_tok = hax.arange(sub_pos, start=i)
        out_tok, cache = _jit_decode(attn, x_tok, pos_ids_tok, cache=cache)
        out_chunks.append(out_tok.array)  # collect raw arrays for concat

    # Concatenate along the position axis (axis index 1)
    decoded_arr = jnp.concatenate(out_chunks, axis=1)

    # Assert equality (within numerical tolerance)
    assert_trees_all_close(full_out.array, decoded_arr, atol=1e-4, rtol=1e-4)


def test_attention_decode_matches_full_prefill():
    """Ensure prefill decode matches full-sequence forward pass."""
    B = Axis("batch", 2)
    Pos = Axis("position", 4)
    Embed = Axis("embed", 16)

    # Build a tiny attention module without ROPE so that pos_ids do not affect outputs
    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2

    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    cache = cfg.empty_kv_cache(B, Pos, dtype=jnp.float32)
    pos_ids = hax.arange(Pos, dtype=jnp.int32)
    decode_out, new_cache = _jit_decode(attn, x, pos_ids, cache)

    # Assert equality (within numerical tolerance)
    assert_trees_all_close(full_out.array, decode_out.array, atol=1e-4, rtol=1e-4)


# This is a bottleneck in tests
jit_dpa = equinox.filter_jit(dot_product_attention)


@equinox.filter_jit
def _jit_decode(attn, x, pos_ids, cache):
    return attn.decode(
        x,
        pos_ids=pos_ids,
        key=jrandom.PRNGKey(2),
        kv_cache=cache,
    )


# @equinox.filter_jit
def _jit_paged_decode(attn, x, pos_ids, cache: KvPageCache, binfo: PageBatchInfo) -> tuple[NamedArray, KvPageCache]:
    return attn.paged_decode(x, cache, binfo, pos_ids=pos_ids, key=jrandom.PRNGKey(2))


@pytest.mark.parametrize("prefix_size", [1, 2, 3])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_attention_decode_prefill_in_chunks(prefix_size, chunk_size):
    """Ensure prefill decode matches full-sequence forward pass when decoding in chunks."""
    B = Axis("batch", 2)
    Pos = Axis("position", prefix_size + 4 * chunk_size)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    # prefill the prefix_size
    prefix = x[Pos, 0:prefix_size]
    cache = cfg.empty_kv_cache(B, Pos, dtype=jnp.float32)
    prefill_chunk, cache = _jit_decode(
        attn, prefix, pos_ids=hax.arange(Pos.resize(prefix_size), dtype=jnp.int32), cache=cache
    )

    # ok now decode the rest in chunks
    out_chunks = [prefill_chunk]
    for i in range(prefix_size, Pos.size, chunk_size):
        x_tok = x[Pos, hax.dslice(i, chunk_size)]
        sub_pos = x_tok.resolve_axis("position")
        pos_ids_tok = hax.arange(sub_pos, dtype=jnp.int32, start=i)
        out_tok, cache = _jit_decode(attn, x_tok, pos_ids_tok, cache)
        out_chunks.append(out_tok)

    # Concatenate along the position axis
    decoded_arr = hax.concatenate("position", out_chunks)
    assert_trees_all_close(full_out, decoded_arr, atol=1e-4, rtol=1e-4)


def test_attention_decode_ragged_fill_in_chunks():
    """
    Ensure prefill decode matches full-sequence forward pass when decoding in chunks with ragged fill.

    To test this, we will always do the attention computation in terms of a full Pos, but
    we'll use pos_ids < 0 to control the length.
    """
    B = Axis("batch", 2)
    Pos = Axis("position", 8)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    def padded_pos_ids(start, stop):
        """Generate pos_ids with negative values for padding"""
        pos_ids = hax.arange(Pos, dtype=jnp.int32, start=start)
        # Fill with -1 to indicate unused positions
        return hax.where(pos_ids >= stop, -1, pos_ids)

    cache = cfg.empty_kv_cache(B, Pos, dtype=jnp.float32)

    chunk_sizes = [
        [4, 2],
        [0, 1],
        [0, 1],
        [2, 1],
        [1, 2],
        [1, 1],
    ]

    off0, off1 = 0, 0
    outputs0 = []
    outputs1 = []
    for step0, step1 in chunk_sizes:
        pos_ids = hax.stack("batch", [padded_pos_ids(off0, off0 + step0), padded_pos_ids(off1, off1 + step1)])

        # grab the right parts of x
        x0 = x[B, 0, "position", off0 : off0 + step0]
        x1 = x[B, 1, "position", off1 : off1 + step1]

        x_q = hax.full((B, Pos, Embed), 100, dtype=x.dtype)
        x_q = x_q.at[B, 0, "position", 0:step0].set(x0)
        x_q = x_q.at[B, 1, "position", 0:step1].set(x1)

        output, cache = _jit_decode(attn, x_q, pos_ids=pos_ids, cache=cache)
        # slice the output to match the pos_ids
        outputs0.append(output[B, 0, "position", hax.dslice(0, step0)])
        outputs1.append(output[B, 1, "position", hax.dslice(0, step1)])
        off0 += step0
        off1 += step1

    # Concatenate along the position axis
    outputs0_cat = hax.concatenate("position", outputs0)
    outputs1_cat = hax.concatenate("position", outputs1)

    assert_trees_all_close(full_out[B, 0].array, outputs0_cat.array, atol=1e-4, rtol=1e-4)
    assert_trees_all_close(full_out[B, 1].array, outputs1_cat.array, atol=1e-4, rtol=1e-4)

    decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])

    # Assert equality (within numerical tolerance)
    assert_trees_all_close(full_out.array, decoded_arr.array, atol=1e-4, rtol=1e-4)


def test_attention_paged_decode_matches_full_ar():
    Pos = Axis("position", 4)
    Embed = Axis("embed", 8)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq_id = pt.assign_seq_id_to_seq()
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)
    out_chunks = []
    for i in range(Pos.size):
        pt, binfo = pt.allocate_for_seqs(
            updated_seqs=hax.named([seq_id], "seq"),
            new_counts=hax.named([1], "seq"),
            tokens=hax.named([seq_id], "position"),
        )

        x_tok = x[Pos, hax.dslice(i, 1)]
        sub_pos = x_tok.resolve_axis("position")
        pos_ids_tok = hax.arange(sub_pos, start=i)
        out_tok, kv_cache = _jit_paged_decode(attn, x_tok, pos_ids_tok, kv_cache, binfo)
        out_chunks.append(out_tok.array)

    decoded_arr = jnp.concatenate(out_chunks, axis=0)
    assert_trees_all_close(full_out.array, decoded_arr, atol=1e-4, rtol=1e-4)


def test_attention_paged_decode_matches_full_prefill():
    Pos = Axis("position", 16)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()

    x = hax.random.normal(x_key, (Pos, Embed)) * 0.2
    seq_ids = hax.named([seq1, seq2, -1, -1, -1, -1, -1, -1], "seq")
    new_token_counts = hax.named([4, 3, 0, 0, 0, 0, 0, 0], "seq")

    seg_ids = hax.named([0] * 4 + [1] * 3 + [-1] * 9, "position")
    pt, binfo = pt.allocate_for_seqs(updated_seqs=seq_ids, new_counts=new_token_counts, tokens=seg_ids)

    causal = AttentionMask.causal().with_segment_ids(seg_ids)
    full_out = attn(x, causal, key=jrandom.PRNGKey(1))

    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    pos_ids = hax.arange(Pos, dtype=jnp.int32)
    decode_out, _ = _jit_paged_decode(attn, x, pos_ids, kv_cache, binfo)

    # we only care about the first 7 positions, since the rest are padding
    full_out = full_out["position", hax.dslice(0, 7)]
    decode_out = decode_out["position", hax.dslice(0, 7)]

    assert_trees_all_close(full_out.array, decode_out.array, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("prefix_size", [1, 2, 3])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_attention_paged_decode_prefill_in_chunks(prefix_size, chunk_size):
    Pos = Axis("position", prefix_size + 4 * chunk_size)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, rope=None, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)

    B = Axis("batch", 2)
    # x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2
    x = hax.arange((B, Pos, Embed), start=-2, step=0.1, dtype=jnp.float32)
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    seq_axis = Axis("seq", 2)
    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    x0 = x[B, 0]
    x1 = x[B, 1]

    outputs0 = []
    outputs1 = []

    # prefill
    updated = hax.named([seq1, seq2], seq_axis)
    new_counts = hax.named([prefix_size, prefix_size], seq_axis)
    tok_axis = Axis("position", 2 * prefix_size)
    tokens = hax.named([seq1] * prefix_size + [seq2] * prefix_size, tok_axis)
    pt, binfo = pt.allocate_for_seqs(updated, new_counts, tokens)
    x_prefill = hax.concatenate(
        "position",
        [x0[Pos, 0:prefix_size], x1[Pos, 0:prefix_size]],
    )
    pos_ids_prefill = hax.named(list(range(prefix_size)) + list(range(prefix_size)), tok_axis)
    out, kv_cache = _jit_paged_decode(attn, x_prefill, pos_ids_prefill, kv_cache, binfo)
    outputs0.append(out["position", hax.dslice(0, prefix_size)])
    outputs1.append(out["position", hax.dslice(prefix_size, prefix_size)])

    # decode rest in chunks
    for i in range(prefix_size, Pos.size, chunk_size):
        updated = hax.named([seq1, seq2], seq_axis)
        new_counts = hax.named([chunk_size, chunk_size], seq_axis)
        tok_axis = Axis("position", 2 * chunk_size)
        tokens = hax.named([seq1] * chunk_size + [seq2] * chunk_size, tok_axis)
        pt, binfo = pt.allocate_for_seqs(updated, new_counts, tokens)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(i, chunk_size)], x1[Pos, hax.dslice(i, chunk_size)]],
        )
        pos_ids_chunk = hax.named(
            list(range(i, i + chunk_size)) + list(range(i, i + chunk_size)),
            tok_axis,
        )
        out_chunk, kv_cache = _jit_paged_decode(attn, x_chunk, pos_ids_chunk, kv_cache, binfo)
        outputs0.append(out_chunk["position", hax.dslice(0, chunk_size)])
        outputs1.append(out_chunk["position", hax.dslice(chunk_size, chunk_size)])

    outputs0_cat = hax.concatenate("position", outputs0)
    outputs1_cat = hax.concatenate("position", outputs1)
    decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])
    assert_trees_all_close(full_out.array, decoded_arr.array, atol=1e-4, rtol=1e-4)


def test_attention_paged_decode_ragged_fill_in_chunks():
    B = Axis("batch", 2)
    Pos = Axis("position", 8)
    Embed = Axis("embed", 16)

    cfg = AttentionConfig(Embed=Embed, num_heads=2, num_kv_heads=2, attn_backend=AttentionBackend.VANILLA)
    attn_key, x_key = jrandom.split(jrandom.PRNGKey(0))
    attn = Attention.init(cfg, key=attn_key)
    # x = hax.random.normal(x_key, (B, Pos, Embed)) * 0.2
    x = hax.arange((B, Pos, Embed), start=-2, step=0.1, dtype=jnp.float32)
    full_out = attn(x, AttentionMask.causal(), key=jrandom.PRNGKey(1))

    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    x0 = x[B, 0]
    x1 = x[B, 1]

    chunk_sizes = [[4, 2], [0, 1], [0, 1], [2, 1], [1, 2], [1, 1]]
    off0 = off1 = 0
    outputs0 = []
    outputs1 = []

    seq_axis = Axis("seq", 2)
    for step0, step1 in chunk_sizes:
        tok_axis = Axis("position", step0 + step1)
        updated = hax.named([seq1, seq2], seq_axis)
        new_counts = hax.named([step0, step1], seq_axis)
        tokens = hax.named([seq1] * step0 + [seq2] * step1, tok_axis)
        pt, binfo = pt.allocate_for_seqs(updated, new_counts, tokens)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(off0, step0)], x1[Pos, hax.dslice(off1, step1)]],
        )
        pos_ids = hax.named(
            list(range(off0, off0 + step0)) + list(range(off1, off1 + step1)),
            tok_axis,
        )
        with jax.disable_jit():
            output, kv_cache = _jit_paged_decode(attn, x_chunk, pos_ids=pos_ids, cache=kv_cache, binfo=binfo)
        print(off0, off1, step0, step1)
        outputs0.append(output["position", hax.dslice(0, step0)])
        outputs1.append(output["position", hax.dslice(step0, step1)])

        # check each chunk individually
        assert_trees_all_close(
            full_out[B, 0, "position", hax.dslice(off0, step0)].array,
            outputs0[-1].array,
            atol=1e-4,
            rtol=1e-4,
        )
        assert_trees_all_close(
            full_out[B, 1, "position", hax.dslice(off1, step1)].array,
            outputs1[-1].array,
            atol=1e-4,
            rtol=1e-4,
        )

        off0 += step0
        off1 += step1

    outputs0_cat = hax.concatenate("position", outputs0)
    outputs1_cat = hax.concatenate("position", outputs1)

    decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])
    assert_trees_all_close(full_out.array, decoded_arr.array, atol=1e-4, rtol=1e-4)
