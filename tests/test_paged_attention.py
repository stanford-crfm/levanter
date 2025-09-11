import haliax as hax
import jax
import pytest
from chex import assert_trees_all_close
from haliax import NamedArray, Axis
from jax import random as jrandom, numpy as jnp

from levanter.inference.page_table import PageBatchInfo, PageTable
from levanter.layers import AttentionConfig, AttentionBackend, Attention, AttentionMask
from levanter.layers.attention import KvPageCache


def _jit_paged_decode(attn, x, pos_ids, cache: KvPageCache, binfo: PageBatchInfo) -> tuple[NamedArray, KvPageCache]:
    return attn.paged_decode(x, cache, binfo, pos_ids=pos_ids, key=jrandom.PRNGKey(2))


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
        # Compute pos_ids for this allocation using current seq_lens before allocation
        seg_ids = hax.named([seq_id], "position")
        # relative position inside this seg is 0 for this single token; absolute pos is current len
        abs_pos = pt.seq_lens["seq", seg_ids].array
        pos_ids = hax.named(abs_pos, "position")

        pt, binfo = pt.allocate_for_seq(seg_ids)

        x_tok = x[Pos, hax.dslice(i, 1)]
        out_tok, kv_cache = _jit_paged_decode(attn, x_tok, pos_ids, kv_cache, binfo)
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
    seg_ids = hax.named([0] * 4 + [1] * 3 + [-1] * 9, "position")
    pt, binfo = pt.allocate_for_seq(seg_ids)

    causal = AttentionMask.causal().with_segment_ids(seg_ids)
    full_out = attn(x, causal, key=jrandom.PRNGKey(1))

    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    # Compute absolute pos ids for this batch from current seq_lens
    def _relative_positions(seg_ids):
        idx = jnp.arange(seg_ids.shape[0])
        is_start = jnp.concatenate([jnp.array([True]), seg_ids[1:] != seg_ids[:-1]])
        start_idx = idx * is_start.astype(idx.dtype)
        seg_start = jax.lax.associative_scan(jnp.maximum, start_idx)
        return idx - seg_start

    rel_pos = _relative_positions(seg_ids.array)
    starts = pt.seq_lens["seq", seg_ids].array
    pos_ids = hax.named(starts + rel_pos, "position")

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

    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()
    kv_cache = attn.empty_page_cache(pt, dtype=jnp.float32)

    x0 = x[B, 0]
    x1 = x[B, 1]

    outputs0 = []
    outputs1 = []

    # prefill
    tok_axis = Axis("position", 2 * prefix_size)
    tokens = hax.named([seq1] * prefix_size + [seq2] * prefix_size, tok_axis)
    pt, binfo = pt.allocate_for_seq(tokens)
    x_prefill = hax.concatenate(
        "position",
        [x0[Pos, 0:prefix_size], x1[Pos, 0:prefix_size]],
    )
    # compute pos ids for prefill chunk from current seq_lens
    def _relative_positions(seg_ids):
        idx = jnp.arange(seg_ids.shape[0])
        is_start = jnp.concatenate([jnp.array([True]), seg_ids[1:] != seg_ids[:-1]])
        start_idx = idx * is_start.astype(idx.dtype)
        seg_start = jax.lax.associative_scan(jnp.maximum, start_idx)
        return idx - seg_start

    rel_pos = _relative_positions(tokens.array)
    starts = pt.seq_lens["seq", tokens].array
    pos_ids = hax.named(starts + rel_pos, "position")
    out, kv_cache = _jit_paged_decode(attn, x_prefill, pos_ids, kv_cache, binfo)
    outputs0.append(out["position", hax.dslice(0, prefix_size)])
    outputs1.append(out["position", hax.dslice(prefix_size, prefix_size)])

    # decode rest in chunks
    for i in range(prefix_size, Pos.size, chunk_size):
        tok_axis = Axis("position", 2 * chunk_size)
        tokens = hax.named([seq1] * chunk_size + [seq2] * chunk_size, tok_axis)
        pt, binfo = pt.allocate_for_seq(tokens)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(i, chunk_size)], x1[Pos, hax.dslice(i, chunk_size)]],
        )
        rel_pos = _relative_positions(tokens.array)
        starts = pt.seq_lens["seq", tokens].array
        pos_ids = hax.named(starts + rel_pos, "position")
        out_chunk, kv_cache = _jit_paged_decode(attn, x_chunk, pos_ids, kv_cache, binfo)
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

    for step0, step1 in chunk_sizes:
        tok_axis = Axis("position", step0 + step1)
        seg_ids = hax.named([seq1] * step0 + [seq2] * step1, tok_axis)
        pt, binfo = pt.allocate_for_seq(seg_ids)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(off0, step0)], x1[Pos, hax.dslice(off1, step1)]],
        )
        # compute pos ids for this ragged chunk
        def _relative_positions(seg_ids):
            idx = jnp.arange(seg_ids.shape[0])
            is_start = jnp.concatenate([jnp.array([True]), seg_ids[1:] != seg_ids[:-1]])
            start_idx = idx * is_start.astype(idx.dtype)
            seg_start = jax.lax.associative_scan(jnp.maximum, start_idx)
            return idx - seg_start

        rel_pos = _relative_positions(seg_ids.array)
        starts = pt.seq_lens["seq", seg_ids].array
        pos_ids = hax.named(starts + rel_pos, "position")
        output, kv_cache = _jit_paged_decode(attn, x_chunk, pos_ids=pos_ids, cache=kv_cache, binfo=binfo)
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
