# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import haliax as hax

from haliax import Axis

import numpy as np

from levanter.inference.engine import InferenceEngine, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.layers.attention import KvPageCache
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID
import pytest
import logging


class DummyModel:
    """Minimal model stub to drive GenerationService for tests.

    - `initial_cache` returns an empty KvPageCache sized to the PageTable.
    - `decode` returns constant logits that strongly prefer token `EOS`.
    """

    def __init__(self, vocab_size: int, eos_id: int = 3):
        self.Vocab = Axis("vocab", vocab_size)
        self.eos = eos_id

    def initial_cache(self, page_table: PageTable, *, dtype):
        # Use trivial cache dimensions; the cache is unused by this dummy model
        kv_heads = Axis("kv_head", 1)
        head_size = Axis("embed", 1)
        return KvPageCache.init(page_table, kv_heads, head_size, dtype=dtype)

    def decode(self, input_ids, kv_cache, batch_info, pos_ids):
        # Produce logits that prefer `eos` for every sampled position
        Pos = input_ids.resolve_axis("position")
        Vocab = self.Vocab
        # One-hot on vocab axis for eos token, broadcast over positions
        logits = hax.nn.one_hot(self.eos, Vocab, dtype=jnp.float32)
        logits = logits.broadcast_axis(Pos)
        return logits, kv_cache


def _build_service(vocab_size=10):
    model = DummyModel(vocab_size=vocab_size, eos_id=3)
    service = InferenceEngine.from_model(
        model=model,
        tokenizer=None,
        max_pages=64,
        max_seqs=8,
        page_size=8,
        max_pages_per_seq=4,
        compute_dtype=jnp.float32,
        max_queued_tokens=64,
        max_seqs_in_prefill=4,
    )
    return service


def test_release_on_finish_and_reuse_slots(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO)
    svc = _build_service()

    prompts = [[7, 7], [1]]
    stop_tokens = [3]  # Dummy model always emits 3, so immediate stop after first generated token

    # Build Requests for the new API
    stop_ids = hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("position",)).broadcast_axis({"stop_seq": 1})
    reqs = []
    for i, toks in enumerate(prompts):
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(toks) + 5, dtype=jnp.int32),
            stop_tokens=stop_ids,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(i),
        )
        reqs.append(Request(prompt_tokens=toks, request_id=i, decode_params=seq_params, n_generations=1))

    outputs, total_generated = svc.generate(reqs)

    # Each sequence should be original prompt + a single eos token
    # TODO: we recently stopped appending prompt to outputs; re-enable these checks if we restore that behavior
    # assert outputs[0] == prompts[0] + [3]
    # assert outputs[1] == prompts[1] + [3]
    assert outputs[0] == [3]
    assert outputs[1] == [3]
    assert total_generated == 2  # one new token per prompt

    # Finished sequences are auto-released; PageTable should have no active seqs
    pt = svc.gen_state.decode_state.page_table
    ds = svc.gen_state.decode_state
    # All slots should be marked unused and lengths zeroed
    seq_lens = jax.device_get(pt.seq_lens.array)
    used_mask = jax.device_get(pt.used_mask.array)
    assert (used_mask == 0).all()
    assert (seq_lens == 0).all()
    # All clone sources should be INVALID
    clone_sources = jax.device_get(ds.clone_sources.array)
    assert (clone_sources == INVALID).all()
    # No pages should be held
    ref_counts = jax.device_get(pt.page_ref_counts.array)
    assert int(ref_counts.sum()) == 0

    # Ensure we logged the release
    assert any("Releasing finished sequences" in rec.message for rec in caplog.records)

    # Now reuse service for another prompt without calling reset()
    prompts2 = [[5, 5, 5]]
    # Build a second set of Requests to ensure reuse works without reset()
    stop_ids2 = hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("position",)).broadcast_axis(
        {"stop_seq": 1}
    )
    reqs2 = []
    for i, toks in enumerate(prompts2):
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(toks) + 3, dtype=jnp.int32),
            stop_tokens=stop_ids2,
            temperature=jnp.array(0.0, dtype=jnp.float32),
            key=jax.random.PRNGKey(42 + i),
        )
        reqs2.append(Request(prompt_tokens=toks, request_id=i, decode_params=seq_params, n_generations=1))

    outputs2, total_generated2 = svc.generate(reqs2)
    # TODO: re-enable if we restore prompt prepending
    # assert outputs2[0] == prompts2[0] + [3]
    assert outputs2[0] == [3]
    assert total_generated2 == 1


def test_reuse_with_clones_and_slot_reassignment():
    svc = _build_service()

    prompts = [[7, 7], [1, 2]]
    stop_tokens = [3]

    stop_ids = hax.named(jnp.asarray(stop_tokens, dtype=jnp.int32), axis=("position",)).broadcast_axis({"stop_seq": 1})

    def build_requests(seed_offset: int) -> list[Request]:
        reqs: list[Request] = []
        for i, toks in enumerate(prompts):
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(len(toks) + 5, dtype=jnp.int32),
                stop_tokens=stop_ids,
                temperature=jnp.array(0.0, dtype=jnp.float32),
                key=jax.random.PRNGKey(seed_offset + i),
            )
            reqs.append(Request(prompt_tokens=toks, request_id=i, decode_params=seq_params, n_generations=3))
        return reqs

    reqs = build_requests(0)
    outputs, total_generated = svc.generate(reqs)
    assert all(out == [3] for out in outputs)
    assert total_generated == len(outputs)

    reqs2 = build_requests(100)
    outputs2, total_generated2 = svc.generate(reqs2)
    assert all(out == [3] for out in outputs2)
    assert total_generated2 == len(outputs2)


def test_page_table_allocation_unsorted_slots():
    page_table = PageTable.init(max_pages=32, max_seqs=8, page_size=8, max_pages_per_seq=2)

    # Mark the target sequence slots as used to mirror runtime behavior
    for seq in (6, 7):
        page_table, _ = page_table.assign_seq_id_to_seq(seq)

    slot_ids_desc = hax.named(jnp.asarray([7, 7, 6, 6], dtype=jnp.int32), axis=("position",))

    new_table, batch_desc = page_table.allocate_for_seq(token_slot_ids=slot_ids_desc)

    seq_lens = np.asarray(jax.device_get(new_table.seq_lens.array))
    assert seq_lens[7] == 2
    assert seq_lens[6] == 2

    dests = np.asarray(jax.device_get(batch_desc.new_token_dests.array))[:4]
    perm = np.asarray(jax.device_get(batch_desc.token_permutation.array))[:4]
    sorted_slots = np.asarray(jax.device_get(slot_ids_desc.array))[perm]
    sorted_dests = dests

    for seq in (6, 7):
        seq_dests = sorted_dests[sorted_slots == seq]
        assert np.all(seq_dests[:-1] <= seq_dests[1:])
