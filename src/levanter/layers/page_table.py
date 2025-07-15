import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.haxtyping as ht
from haliax import NamedArray

__all__ = ["PageTable", "PageBatchInfo"]


def _relative_positions(seg_ids: jnp.ndarray):
    idx = jnp.arange(seg_ids.shape[0])
    is_start = jnp.concatenate([jnp.array([True]),
                                seg_ids[1:] != seg_ids[:-1]])
    start_idx = idx * is_start.astype(idx.dtype)
    seg_start = jnp.maximum.accumulate(start_idx)
    return idx - seg_start  # 0,1,2,… inside each segment


class PageTable(eqx.Module):
    """Tracks which pages are allocated to which sequences."""

    page_indices: NamedArray  # i32[Seq, PagePerSeq]
    page_owners: NamedArray  # i32[Page]
    seq_lens: NamedArray  # i32[Seq]
    page_size: int = eqx.field(static=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_pages(self) -> int:
        return self.page_indices.axis_size("page") * self.page_indices.axis_size("seq")

    @property
    def pages_per_seq(self) -> int:
        return self.page_indices.axis_size("page")

    @property
    def max_seqs(self) -> int:
        return self.page_indices.axis_size("seq")

    @property
    def max_len_per_seq(self) -> int:
        return self.page_size * self.pages_per_seq

    @property
    def current_num_seqs(self) -> int:
        return hax.sum(self.seq_lens >= 0).scalar()

    @property
    def max_Seq(self) -> hax.Axis:
        """Axis representing the maximum number of sequences."""
        return hax.Axis("seq", self.max_seqs)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @staticmethod
    def init(max_pages: int, max_seqs: int, page_size: int, max_pages_per_seq: int) -> "PageTable":
        page_indices = hax.full({"seq": max_seqs, "page": max_pages_per_seq}, -1, dtype=jnp.int32)
        page_owners = hax.full({"page": max_pages}, -1, dtype=jnp.int32)
        seq_lens = hax.full({"seq": max_seqs}, -1, dtype=jnp.int32)
        return PageTable(page_indices, page_owners, seq_lens, page_size)

    # ------------------------------------------------------------------
    # Sequence management
    # ------------------------------------------------------------------
    @eqx.filter_jit(donate="all")
    def assign_seq_id_to_seq(self) -> tuple["PageTable", int]:
        seq_id = hax.argmin(self.seq_lens, "seq").scalar()
        # Error handling: if there are no sequences available, return -1
        seq_id = hax.where(self.seq_lens["seq", seq_id] < 0, seq_id, -1)
        new_seq_lens = hax.where(
            seq_id >= 0,
            self.seq_lens.at["seq", seq_id].set(0),
            self.seq_lens)
        return dataclasses.replace(self, seq_lens=new_seq_lens), seq_id

    def allocate_for_seq(
        self,
        token_seq_ids: ht.i32[NamedArray, " position"],  # type: ignore[name-defined]
    ) -> tuple["PageTable", "PageBatchInfo"]:
        """Allocate pages for new sequences and update ``seq_lens``."""

        page_indices = self.page_indices
        page_owners = self.page_owners
        seq_lens = self.seq_lens

        token_seq_ids = hax.where(token_seq_ids < 0, self.max_seqs, token_seq_ids)
        updated_seqs, new_counts = hax.unique_counts(token_seq_ids, self.max_Seq, fill_value=self.max_seqs)

        new_counts = hax.where(updated_seqs >= self.max_seqs, 0, new_counts)

        current_lens = hax.where(seq_lens < 0, 0, seq_lens)
        new_lens = current_lens.at["seq", updated_seqs].add(new_counts, mode="drop")
        # anything that was -1 should still be -1
        new_lens = hax.where(self.seq_lens >= 0, new_lens, -1)

        new_num_pages_needed = (new_lens + self.page_size - 1) // self.page_size
        old_num_pages_needed = (seq_lens + self.page_size - 1) // self.page_size

        def _alloc_pages_for_seq(seq_id, carry):
            page_indices, page_owners = carry
            num_needed = new_num_pages_needed["seq", seq_id].scalar()
            old_needed = old_num_pages_needed["seq", seq_id].scalar()

            def body(page_idx, state):
                page_indices, page_owners = state
                free_page_idx = hax.argmin(page_owners, "page")
                page_owners = page_owners.at["page", free_page_idx].set(seq_id)
                page_indices = page_indices.at["seq", seq_id, "page", page_idx].set(free_page_idx)
                return page_indices, page_owners

            new_page_indices, new_page_owners = jax.lax.fori_loop(
                old_needed, num_needed, body, (page_indices, page_owners)
            )
            return new_page_indices, new_page_owners

        page_indices, page_owners = jax.lax.fori_loop(
            0, self.max_seqs, _alloc_pages_for_seq, (page_indices, page_owners)
        )

        new_table = dataclasses.replace(
            self,
            page_indices=page_indices,
            page_owners=page_owners,
            seq_lens=new_lens,
        )

        batch_info = self._slice_batch_info(updated_seqs, self.seq_lens, new_table, new_counts, token_seq_ids)

        return new_table, batch_info

    def _slice_batch_info(self, updated_seqs, old_seq_lens, new_table, new_token_counts, tokens):
        mask = hax.logical_and(updated_seqs >= 0, updated_seqs < self.max_seqs)
        safe_updated = hax.where(mask, updated_seqs, 0)

        gathered_page_indices = new_table.page_indices["seq", safe_updated]
        page_indices = hax.where(mask, gathered_page_indices, -1)

        seq_lens = new_table.seq_lens["seq", safe_updated]
        seq_lens = hax.where(mask, seq_lens, -1)

        num_seqs = hax.sum(mask).scalar()

        token_dests = hax.full(tokens.shape, -1, dtype=jnp.int32)
        seq_cursors = jnp.where(old_seq_lens.array < 0, 0, old_seq_lens.array)

        def token_body(i, carry):
            token_dests, seq_cursors = carry
            seq_id = tokens["position", i].scalar()

            def assign(carry):
                token_dests, seq_cursors = carry
                page_idx = seq_cursors[seq_id] // self.page_size
                page_offset = seq_cursors[seq_id] % self.page_size
                page = new_table.page_indices["seq", seq_id, "page", page_idx]
                dest = hax.where(page < 0, -1, page * self.page_size + page_offset)
                token_dests = token_dests.at["position", i].set(dest)
                seq_cursors = seq_cursors.at[seq_id].add(1)
                return token_dests, seq_cursors

            token_dests, seq_cursors = jax.lax.cond(seq_id >= 0, assign, lambda c: c, (token_dests, seq_cursors))
            return token_dests, seq_cursors

        token_dests, _ = jax.lax.fori_loop(0, tokens.axis_size("position"), token_body, (token_dests, seq_cursors))

        cu_q_lens = hax.concatenate(
            "seq",
            [
                hax.zeros({"seq": 1}, dtype=jnp.int32),
                hax.cumsum(new_token_counts, "seq", dtype=jnp.int32),
            ],
        )
        pos_ids = self.pos_ids_from_seq_ids(tokens)
        batch_info = PageBatchInfo(
            page_indices=page_indices,
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            num_seqs=num_seqs,
            new_token_dests=token_dests,
            pos_ids=pos_ids,
            page_size=self.page_size,
        )
        return batch_info

    @eqx.filter_jit(donate="all")
    def free_pages(self, seq_id: int) -> "PageTable":
        new_page_owners = hax.where(self.page_owners == seq_id, -1, self.page_owners)
        new_page_indices = self.page_indices.at["seq", seq_id].set(-1)
        new_seq_lens = self.seq_lens.at["seq", seq_id].set(-1)

        return dataclasses.replace(
            self,
            page_owners=new_page_owners,
            page_indices=new_page_indices,
            seq_lens=new_seq_lens,
        )

    def pos_ids_from_seq_ids(self, seq_ids: ht.i32[NamedArray, "position"]) -> ht.i32[NamedArray, "position"]:  # type: ignore[name-defined]
        """
        Given sequence IDs, compute the position IDs for each sequence.

        seg_ids may be padded with negative numbers, which will be ignored in the output.
        """
        rel_pos = _relative_positions(seq_ids.array)
        # We need to add the start position of the segment to the relative position
        seg_pos_starts = self.seq_lens["seq", seq_ids].array

        pos_ids = seg_pos_starts + rel_pos
        # mask out the -1 segments
        pos_ids = jnp.where(seq_ids.array < 0, -1, pos_ids)

        return hax.named(pos_ids, "position")


class PageBatchInfo(eqx.Module):
    """Page and length information for a batch of sequences."""

    page_indices: ht.i32[NamedArray, " seq page"]
    seq_lens: ht.i32[NamedArray, " seq"]
    cu_q_lens: ht.i32[NamedArray, " seq"]
    num_seqs: ht.i32[jnp.ndarray, ""]
    new_token_dests: ht.i32[NamedArray, "position"]
    pos_ids: ht.i32[NamedArray, "position"]
    page_size: int = eqx.field(static=True)

    def __post_init__(self):
        assert isinstance(self.num_seqs, jnp.ndarray), "num_seqs must be a JAX ndarray"
