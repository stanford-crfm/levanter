import dataclasses

import jax.numpy as jnp

import haliax as hax

from levanter.layers.attention import PageBatchInfo, PageTable


def _make_table(pages=8, seqs=4, page_size=2, pages_per_seq=2):
    return PageTable.init(pages, seqs, page_size, pages_per_seq)


def test_page_table_max_len_per_seq():
    pt = _make_table(page_size=2, pages_per_seq=3)
    assert pt.max_len_per_seq == 6


def test_page_table_free_pages():
    pt = _make_table()
    # manually set some ownership
    pt = dataclasses.replace(
        pt,
        page_owners=hax.named(jnp.array([0, 0, -1, -1, -1, -1, -1, -1], dtype=jnp.int32), "page"),
        page_indices=pt.page_indices.at["seq", 0, "page", 0].set(0).at["seq", 0, "page", 1].set(1),
        seq_lens=pt.seq_lens.at["seq", 0].set(4),
    )

    freed = PageTable.free_pages(pt, 0)

    assert jnp.all(freed.page_owners.array[:2] == -1)
    assert jnp.all(freed.page_indices.array[0] == -1)
    assert freed.seq_lens.array[0] == -1


def test_page_batch_info_shapes():
    seq = hax.Axis("seq", 2)
    page = hax.Axis("page", 3)
    pb = PageBatchInfo(
        page_indices=hax.full((seq, page), -1, dtype=jnp.int32),
        seq_lens=hax.full((seq,), -1, dtype=jnp.int32),
        cu_q_lens=jnp.array([0, 1, 2], dtype=jnp.int32),
        num_seqs=jnp.array(2, dtype=jnp.int32),
        new_token_dests=jnp.array([-1, -1], dtype=jnp.int32),
        page_size=2,
    )

    assert pb.page_indices.axes == (seq, page)
    assert pb.seq_lens.axes == (seq,)
    assert pb.cu_q_lens.shape[0] == pb.num_seqs + 1
