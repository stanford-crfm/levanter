# tests/test_ragged_paged_attention.py
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from chex import assert_trees_all_close

import haliax as hax
import haliax.nn as hnn

from levanter.layers.attention import AttentionMask, default_ragged_paged_attention


PAGE = hax.Axis("page", 128)  # plenty of slack
SLOT = hax.Axis("slot", 4)  # page size
KV_HEADS = hax.Axis("kv_heads", 4)
QH = hax.Axis("q_heads_per_group", 1)
D = hax.Axis("head_size", 32)


KV_BS = 32  # must match constant inside kernel
SM_SCALE = 1 / math.sqrt(D.size)


def _build_random_case(rng, seq_lens):
    """Seq‑by‑seq contiguous paging with padding to KV_BS."""
    num_seqs = len(seq_lens)

    # token → sequence mapping and cumulative Q lengths
    tok_offsets = np.cumsum([0] + seq_lens)
    cu_q_lens = jnp.asarray(tok_offsets, dtype=jnp.int32)

    # allocate enough pages for the longest sequence, but give each sequence its own slice
    pages_per_seq_raw = [(length + SLOT.size - 1) // SLOT.size for length in seq_lens]
    max_pages = KV_BS * ((max(pages_per_seq_raw) + KV_BS - 1) // KV_BS)  # pad to mult of 32
    page_indices = -jnp.ones((num_seqs, max_pages), dtype=jnp.int32)

    # generate actual page contents
    next_free_page = 0
    kv_data = []
    for sid, (qlen, p_raw) in enumerate(zip(seq_lens, pages_per_seq_raw)):
        pages = jnp.arange(next_free_page, next_free_page + p_raw, dtype=jnp.int32)
        page_indices = page_indices.at[sid, :p_raw].set(pages)
        next_free_page += p_raw

        # fill those pages with random k/v
        kv_pages_for_seq = hax.random.normal(
            jr.fold_in(rng, sid),
            (hax.Axis("page", p_raw), SLOT, KV_HEADS.resize(2 * KV_HEADS.size), D),
        )
        kv_data.append(kv_pages_for_seq)

    # stack pages, pad out to PAGE axis
    kv_pages = hax.zeros((PAGE, SLOT, KV_HEADS.resize(2 * KV_HEADS.size), D))
    kv_pages = kv_pages.at["page", :next_free_page].set(hax.concatenate("page", kv_data))

    # random queries for whole token axis
    total_tokens = tok_offsets[-1]
    this_TOK = hax.Axis("tok", int(total_tokens))
    q = hax.random.normal(rng, (this_TOK, KV_HEADS, QH, D)) * SM_SCALE

    kv_lens = jnp.asarray(seq_lens, dtype=jnp.int32)
    kv_lens = hax.named(kv_lens, "seq")

    page_indices = hax.named(page_indices, ("seq", "page"))

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, jnp.array(num_seqs, dtype=jnp.int32)


def _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens):
    """Naïve per‑sequence causal soft‑max (slow but tiny)."""
    out_chunks = []
    for sid, qlen in enumerate(seq_lens):
        # slice query tokens for this sequence
        start = int(cu_q_lens[sid])
        q_seq = q["tok", hax.ds(start, qlen)]
        TOK_S = hax.Axis("Tok_s", qlen)

        # gather kv for the sequence
        n_pages = (kv_lens[sid] + SLOT.size - 1) // SLOT.size
        pages = page_indices[sid, :n_pages]
        kv_flat = kv_pages["page", pages, "slot", :].flatten("kv_tok")
        kv_flat = kv_flat["kv_tok", hax.ds(0, kv_lens[sid])]

        k_seq = kv_flat["kv_heads", : KV_HEADS.size]
        v_seq = kv_flat["kv_heads", KV_HEADS.size :]

        # rename axes so they line up with dot_product_attention sig
        q_seq = q_seq.rename_axis("tok", TOK_S.name)
        k_seq = k_seq.rename_axis("kv_tok", TOK_S.name)
        v_seq = v_seq.rename_axis("kv_tok", TOK_S.name)

        mask = AttentionMask.causal().materialize(TOK_S, TOK_S)
        ref = hnn.attention.dot_product_attention(TOK_S, D, q_seq, k_seq, v_seq, mask=mask)

        out_chunks.append(ref.rename({TOK_S.name: "tok"}))

    return hax.concatenate("tok", out_chunks)


# ---------------------------------------------------------------------------
#                              actual tests
# ---------------------------------------------------------------------------


def test_ragged_paged_attention_single_seq():
    rng = jr.PRNGKey(0)
    seq_lens = [45]  # one ragged sequence
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = default_ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    assert_trees_all_close(ragged.array, ref.array, atol=1e-3, rtol=1e-3)


def test_ragged_paged_attention_multi_seq():
    rng = jr.PRNGKey(1)
    seq_lens = [10, 37, 64]  # three different lengths
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = default_ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    assert_trees_all_close(ragged.array, ref.array, atol=1e-3, rtol=1e-3)
