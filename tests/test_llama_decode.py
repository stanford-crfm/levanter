import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from chex import assert_trees_all_close

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask, KvPageState, PageTable, AttentionBackend
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
