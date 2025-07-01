# tests/test_ragged_paged_attention.py
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from chex import assert_trees_all_close

import haliax as hax

from levanter.layers.attention import AttentionMask, default_ragged_paged_attention, simple_attention_with_dropout


PAGE = hax.Axis("page", 64)  # plenty of slack
SLOT = hax.Axis("slot", 4)  # page size
KV_HEADS = hax.Axis("kv_heads", 2)
QH = hax.Axis("q_heads_per_group", 1)
D = hax.Axis("head_size", 4)


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
    q = hax.random.normal(rng, (this_TOK, KV_HEADS, QH, D))

    kv_lens = jnp.asarray(seq_lens, dtype=jnp.int32)
    kv_lens = hax.named(kv_lens, "seq")

    page_indices = hax.named(page_indices, ("seq", "page"))

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, jnp.array(num_seqs, dtype=jnp.int32)


def _build_incremental_case(rng, seq_lens, k_lens):
    """Like ``_build_random_case`` but query only contains the last ``k`` tokens.

    ``seq_lens`` gives the total tokens already in the KV cache for each
    sequence. ``k_lens`` is how many query tokens each sequence has. The KV
    cache still contains ``seq_lens`` tokens for every sequence.
    """
    q_full, kv_pages, kv_lens, page_indices, full_cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    assert len(seq_lens) == len(k_lens)

    chunks = []
    new_offsets = [0]
    for sid, (total_len, k) in enumerate(zip(seq_lens, k_lens)):
        start = int(full_cu_q_lens[sid]) + total_len - k
        chunks.append(q_full["tok", hax.ds(start, k)])
        new_offsets.append(new_offsets[-1] + k)

    q = hax.concatenate("tok", chunks)
    cu_q_lens = jnp.asarray(new_offsets, dtype=jnp.int32)

    return q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs


def _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens):
    """Naïve per‑sequence causal soft‑max (slow but tiny)."""
    out_chunks = []
    for sid, qlen in enumerate(seq_lens):
        # slice query tokens for this sequence
        start = int(cu_q_lens[sid])
        q_seq = q["tok", hax.ds(start, qlen)]
        TOK_S = hax.Axis("tok", qlen)

        # gather kv for the sequence
        n_pages = (kv_lens["seq", sid] + SLOT.size - 1) // SLOT.size
        pages = page_indices["seq", sid, "page", : n_pages.scalar()]
        kv_flat = kv_pages["page", pages, "slot", :].flatten_axes(("page", "slot"), "kv_tok")
        kv_flat = kv_flat["kv_tok", hax.ds(0, kv_lens["seq", sid].scalar())]

        k_seq = kv_flat["kv_heads", : KV_HEADS.size]
        v_seq = kv_flat["kv_heads", KV_HEADS.size :]

        # rename axes so they line up with dot_product_attention sig
        q_seq = q_seq.rename({"tok": TOK_S.name})

        offset = kv_lens["seq", sid].scalar() - qlen
        mask = AttentionMask.causal(offset)
        ref = simple_attention_with_dropout(
            "tok", "kv_tok", D, q_seq, k_seq, v_seq, mask=mask, scaling_factor=SM_SCALE
        )

        out_chunks.append(ref)

    return hax.concatenate("tok", out_chunks)


# ---------------------------------------------------------------------------
#                              actual tests
# ---------------------------------------------------------------------------


def test_ragged_paged_attention_single_seq():
    rng = jr.PRNGKey(0)
    seq_lens = [46]  # one sequence
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = default_ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    assert_trees_all_close(ragged.array[:-1], ref.array[:-1], atol=1e-3, rtol=1e-3)
    assert_trees_all_close(ragged.array[-1], ref.array[-1], atol=1e-3, rtol=1e-3)
    assert_trees_all_close(ragged.array, ref.array, atol=1e-3, rtol=1e-3)


def test_ragged_paged_attention_multi_seq():
    rng = jr.PRNGKey(1)
    seq_lens = [10, 37, 64]  # three different lengths
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_random_case(rng, seq_lens)

    ragged = default_ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, seq_lens)

    assert ragged.axes == ref.axes
    assert_trees_all_close(ragged.array, ref.array, atol=1e-3, rtol=1e-3)


def test_ragged_paged_attention_incremental_single_seq():
    rng = jr.PRNGKey(2)
    seq_lens = [47]
    k_lens = [5]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_incremental_case(rng, seq_lens, k_lens)

    ragged = default_ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, k_lens)

    assert ragged.axes == ref.axes
    assert_trees_all_close(ragged.array, ref.array, atol=1e-3, rtol=1e-3)


def test_ragged_paged_attention_incremental_multi_seq():
    rng = jr.PRNGKey(3)
    seq_lens = [10, 37, 64]
    k_lens = [1, 3, 9]
    q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs = _build_incremental_case(rng, seq_lens, k_lens)

    ragged = default_ragged_paged_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=SM_SCALE)
    ref = _reference_attention(q, kv_pages, kv_lens, page_indices, cu_q_lens, k_lens)

    assert ragged.axes == ref.axes
    assert_trees_all_close(ragged.array, ref.array, atol=1e-3, rtol=1e-3)
