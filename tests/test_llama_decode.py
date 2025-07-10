import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from chex import assert_trees_all_close
import pytest

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask, KvPageState, AttentionBackend
from levanter.layers.page_table import PageTable
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel


@eqx.filter_jit
def _jit_paged_decode(transformer, x, pos_ids, state):
    """Jitted wrapper around ``transformer.decode`` for a single decoding step."""
    return transformer.decode(state, x, pos_ids, key=jrandom.PRNGKey(2))


def test_llama_paged_decode_matches_full_ar():
    """Ensure llama incremental decode matches full-sequence forward pass for activations."""
    # Axes
    Pos = Axis("position", 4)
    Embed = Axis("embed", 8)
    Vocab = Axis("vocab", 64)

    # Minimal Llama config (2 layers, no ROPE, vanilla attention)
    cfg = LlamaConfig(
        seq_len=Pos.size,
        hidden_dim=Embed.size,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        gradient_checkpointing=False,
        scan_layers=True,
        attn_backend=AttentionBackend.VANILLA,
    )

    # Instantiate a tiny model
    model_key, input_key = jrandom.split(jrandom.PRNGKey(0))
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=cfg, key=model_key)

    # Random input ids (no batch axis for simplicity)
    input_ids = hax.random.randint(input_key, Pos, 0, Vocab.size)
    mask = AttentionMask.causal()

    # Full forward pass ------------------------------------------------------------------
    full_out = model.activations(input_ids, attn_mask=mask, key=jrandom.PRNGKey(1))

    # Prepare paged KV cache --------------------------------------------------------------
    pt = PageTable.init(
        max_pages=Pos.size,
        max_seqs=1,
        page_size=Pos.size,
        max_pages_per_seq=Pos.size,
    )
    pt, seq_id = pt.assign_seq_id_to_seq()

    # One cache per layer
    layer_caches = model.transformer.initial_cache(pt, dtype=jnp.float32)

    out_chunks = []
    for i in range(Pos.size):
        # Allocate space for the next token
        pt, binfo = pt.allocate_for_seqs(
            updated_seqs=hax.named([seq_id], "seq"),
            new_counts=hax.named([1], "seq"),
            tokens=hax.named([seq_id], "position"),
        )

        # Wrap per-layer caches in KvPageState referencing the newly allocated pages
        layer_states = KvPageState.from_batch(binfo, layer_caches)

        # Embed the current token
        x_tok_ids = input_ids["position", hax.dslice(i, 1)]
        x_tok = model.embeddings.embed(x_tok_ids)

        sub_pos = x_tok.resolve_axis("position")
        pos_ids_tok = hax.arange(sub_pos, start=i)

        # Decode one step
        out_tok, new_states = _jit_paged_decode(model.transformer, x_tok, pos_ids_tok, layer_states)

        # layer_caches = jax.tree_util.tree_map(lambda state: state.cache, new_states)
        layer_caches = new_states.cache

        out_chunks.append(out_tok.array)

    # Concatenate along the position axis and compare -------------------------------------
    decoded_arr = jnp.concatenate(out_chunks, axis=0)
    assert_trees_all_close(full_out.array, decoded_arr, atol=1e-4, rtol=1e-4)


def test_llama_model_decode_logits():
    """End-to-end check that model.decode reproduces full-forward logits."""

    Pos = Axis("position", 4)
    Embed = Axis("embed", 8)
    Vocab = Axis("vocab", 32)

    cfg = LlamaConfig(
        seq_len=Pos.size,
        hidden_dim=Embed.size,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        gradient_checkpointing=False,
        scan_layers=True,
        attn_backend=AttentionBackend.VANILLA,
    )

    model_key, input_key = jrandom.split(jrandom.PRNGKey(123))
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=cfg, key=model_key)

    input_ids = hax.random.randint(input_key, Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()

    # Reference full-sequence logits
    full_logits = model(input_ids, attn_mask)

    # Prepare paged cache
    pt = PageTable.init(max_pages=Pos.size, max_seqs=1, page_size=Pos.size, max_pages_per_seq=Pos.size)
    pt, seq_id = pt.assign_seq_id_to_seq()
    layer_caches = model.transformer.initial_cache(pt, dtype=jnp.float32)

    gathered_logits = []

    for i in range(Pos.size):
        # Allocate one token's worth of space
        pt, binfo = pt.allocate_for_seqs(
            updated_seqs=hax.named([seq_id], "seq"),
            new_counts=hax.named([1], "seq"),
            tokens=hax.named([seq_id], "position"),
        )

        # Build KvPageState tree matching layers
        layer_states = KvPageState.from_batch(binfo, layer_caches)

        x_tok_ids = input_ids["position", hax.dslice(i, 1)]
        pos_ids_tok = hax.arange(x_tok_ids.resolve_axis("position"), start=i)

        logits_tok, new_states = model.decode(x_tok_ids, layer_states, pos_ids_tok)

        # Update caches
        layer_caches = new_states.cache

        gathered_logits.append(logits_tok.array)

    step_logits = jnp.concatenate(gathered_logits, axis=0)

    assert_trees_all_close(step_logits, full_logits.array, atol=1e-4, rtol=1e-4)


def test_llama_paged_decode_matches_full_prefill():
    """Ensure llama paged decode matches full forward when prefilling entire sequences."""
    Pos = Axis("position", 16)
    Embed = Axis("embed", 8)
    Vocab = Axis("vocab", 64)

    cfg = LlamaConfig(
        seq_len=Pos.size,
        hidden_dim=Embed.size,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        gradient_checkpointing=False,
        scan_layers=True,
        attn_backend=AttentionBackend.VANILLA,
    )

    model_key, input_key = jrandom.split(jrandom.PRNGKey(0))
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=cfg, key=model_key)

    input_ids = hax.random.randint(input_key, Pos, 0, Vocab.size)

    pt = PageTable.init(max_pages=4, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()

    seq_ids = hax.named([seq1, seq2, -1, -1, -1, -1, -1, -1], "seq")
    new_token_counts = hax.named([4, 3, 0, 0, 0, 0, 0, 0], "seq")
    seg_ids = hax.named([0] * 4 + [1] * 3 + [-1] * 9, "position")
    pt, binfo = pt.allocate_for_seqs(updated_seqs=seq_ids, new_counts=new_token_counts, tokens=seg_ids)

    mask = AttentionMask.causal().with_segment_ids(seg_ids)
    full_out = model.activations(input_ids, attn_mask=mask, key=jrandom.PRNGKey(1))

    layer_caches = model.transformer.initial_cache(pt, dtype=jnp.float32)
    page_state = KvPageState.from_batch(binfo, layer_caches)
    pos_ids = hax.arange(Pos, dtype=jnp.int32)
    x = model.embeddings.embed(input_ids)
    decode_out, _ = _jit_paged_decode(model.transformer, x, pos_ids, page_state)

    full_out = full_out["position", hax.dslice(0, 7)]
    decode_out = decode_out["position", hax.dslice(0, 7)]
    assert_trees_all_close(full_out.array, decode_out.array, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("prefix_size", [1, 2, 3])
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_llama_paged_decode_prefill_in_chunks(prefix_size, chunk_size):
    Pos = Axis("position", prefix_size + 4 * chunk_size)
    Embed = Axis("embed", 8)
    Vocab = Axis("vocab", 64)

    cfg = LlamaConfig(
        seq_len=Pos.size,
        hidden_dim=Embed.size,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        gradient_checkpointing=False,
        scan_layers=True,
        attn_backend=AttentionBackend.VANILLA,
    )

    model_key, input_key = jrandom.split(jrandom.PRNGKey(0))
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=cfg, key=model_key)

    B = Axis("batch", 2)
    input_ids = hax.random.randint(input_key, (B, Pos), 0, Vocab.size)
    full_out = model.activations(input_ids, attn_mask=AttentionMask.causal(), key=jrandom.PRNGKey(1))

    seq_axis = Axis("seq", 2)
    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()
    layer_caches = model.transformer.initial_cache(pt, dtype=jnp.float32)

    x = model.embeddings.embed(input_ids)
    x0 = x[B, 0]
    x1 = x[B, 1]

    outputs0 = []
    outputs1 = []

    updated = hax.named([seq1, seq2], seq_axis)
    new_counts = hax.named([prefix_size, prefix_size], seq_axis)
    tok_axis = Axis("position", 2 * prefix_size)
    tokens = hax.named([seq1] * prefix_size + [seq2] * prefix_size, tok_axis)
    pt, binfo = pt.allocate_for_seqs(updated, new_counts, tokens)
    state = KvPageState.from_batch(binfo, layer_caches)
    x_prefill = hax.concatenate("position", [x0[Pos, 0:prefix_size], x1[Pos, 0:prefix_size]])
    pos_ids_prefill = hax.named(list(range(prefix_size)) + list(range(prefix_size)), tok_axis)
    out, state = _jit_paged_decode(model.transformer, x_prefill, pos_ids_prefill, state)
    layer_caches = state.cache
    outputs0.append(out["position", hax.dslice(0, prefix_size)])
    outputs1.append(out["position", hax.dslice(prefix_size, prefix_size)])

    for i in range(prefix_size, Pos.size, chunk_size):
        updated = hax.named([seq1, seq2], seq_axis)
        new_counts = hax.named([chunk_size, chunk_size], seq_axis)
        tok_axis = Axis("position", 2 * chunk_size)
        tokens = hax.named([seq1] * chunk_size + [seq2] * chunk_size, tok_axis)
        pt, binfo = pt.allocate_for_seqs(updated, new_counts, tokens)
        state = KvPageState.from_batch(binfo, layer_caches)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(i, chunk_size)], x1[Pos, hax.dslice(i, chunk_size)]],
        )
        pos_ids_chunk = hax.named(list(range(i, i + chunk_size)) + list(range(i, i + chunk_size)), tok_axis)
        out_chunk, state = _jit_paged_decode(model.transformer, x_chunk, pos_ids_chunk, state)
        layer_caches = state.cache
        outputs0.append(out_chunk["position", hax.dslice(0, chunk_size)])
        outputs1.append(out_chunk["position", hax.dslice(chunk_size, chunk_size)])

    outputs0_cat = hax.concatenate("position", outputs0)
    outputs1_cat = hax.concatenate("position", outputs1)
    decoded_arr = hax.stack("batch", [outputs0_cat, outputs1_cat])
    assert_trees_all_close(full_out.array, decoded_arr.array, atol=1e-4, rtol=1e-4)


def test_llama_paged_decode_ragged_fill_in_chunks():
    B = Axis("batch", 2)
    Pos = Axis("position", 8)
    Embed = Axis("embed", 8)
    Vocab = Axis("vocab", 64)

    cfg = LlamaConfig(
        seq_len=Pos.size,
        hidden_dim=Embed.size,
        intermediate_dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        rope=None,
        gradient_checkpointing=False,
        scan_layers=True,
        attn_backend=AttentionBackend.VANILLA,
    )

    model_key, input_key = jrandom.split(jrandom.PRNGKey(0))
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=cfg, key=model_key)

    input_ids = hax.random.randint(input_key, (B, Pos), 0, Vocab.size)
    full_out = model.activations(input_ids, attn_mask=AttentionMask.causal(), key=jrandom.PRNGKey(1))

    pt = PageTable.init(max_pages=8, max_seqs=2, page_size=4, max_pages_per_seq=4)
    pt, seq1 = pt.assign_seq_id_to_seq()
    pt, seq2 = pt.assign_seq_id_to_seq()
    layer_caches = model.transformer.initial_cache(pt, dtype=jnp.float32)

    x = model.embeddings.embed(input_ids)
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
        state = KvPageState.from_batch(binfo, layer_caches)

        x_chunk = hax.concatenate(
            "position",
            [x0[Pos, hax.dslice(off0, step0)], x1[Pos, hax.dslice(off1, step1)]],
        )
        pos_ids = hax.named(list(range(off0, off0 + step0)) + list(range(off1, off1 + step1)), tok_axis)
        with jax.disable_jit():
            output, state = _jit_paged_decode(model.transformer, x_chunk, pos_ids, state)
        layer_caches = state.cache
        outputs0.append(output["position", hax.dslice(0, step0)])
        outputs1.append(output["position", hax.dslice(step0, step1)])

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
