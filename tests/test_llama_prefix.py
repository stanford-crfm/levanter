import dataclasses
import os
from functools import reduce
from typing import List, Sequence, Union

import chex
import jax
import numpy as np
import torch
import transformers
from jax import random

import haliax as hax
from haliax import Axis, NamedArray
from levanter.compat.hf_checkpoints import RepoRef
from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from tests.test_utils import (
    skip_if_hf_model_not_accessible,
    skip_if_no_torch,
    skip_in_ci,
)
import torch.nn.functional as F

# Additional rotary config import for llama-3 behaviour
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

from haliax.nn import log_softmax
# -----------------------------------------------------------------------------
# Prompt pieces (same strings used in eval_sliding_lm.py)
# -----------------------------------------------------------------------------
#prefix_singleton = "They were careless people, Tom and Daisy – they smashed up"
#suffix_singleton = " things and creatures and then retreated"
prefix_singleton = "France"
suffix_singleton = " Paris"
# -----------------------------------------------------------------------------
# Select which Llama-3 checkpoint to use. Override via env var `LLAMA3_VARIANT`.
# Supported values: "8B" (default) or "1B".
# -----------------------------------------------------------------------------
LLAMA3_VARIANT = os.getenv("LLAMA3_VARIANT", "8B").upper()

_VARIANT_SPECS = {
    "8B": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "hidden_dim": 4096,
        "intermediate_dim": 14336,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
    },
    "1B": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "hidden_dim": 2048,
        "intermediate_dim": 8192,
        "num_layers": 16,
        "num_heads": 32,
        "num_kv_heads": 8,
    },
}

if LLAMA3_VARIANT not in _VARIANT_SPECS:
    raise ValueError(
        f"Unsupported LLAMA3_VARIANT={LLAMA3_VARIANT}; expected one of {list(_VARIANT_SPECS)}"
    )

_VAR = _VARIANT_SPECS[LLAMA3_VARIANT]

MODEL_ID = _VAR["model_id"]
_HIDDEN_DIM = _VAR["hidden_dim"]
_INTERMEDIATE_DIM = _VAR["intermediate_dim"]
_NUM_LAYERS = _VAR["num_layers"]
_NUM_HEADS = _VAR["num_heads"]
_NUM_KV_HEADS = _VAR["num_kv_heads"]

def compute_extraction_prob(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prefix: str,
    suffix: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    return_log_prob: bool = False,
    return_token_log_probs: bool = False,
    return_token_logits: bool = False,
) -> float | tuple[float, float] | tuple[float, float, list[float]] | tuple[float, float, list[float], list[float]]:
    """
    Compute probability that model generates exact suffix given prefix.

    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prefix: The prefix text
        suffix: The target suffix text to check memorization for
        temperature: Temperature for probability scaling (default: 1.0)
        top_k: Apply top-k filtering (default: None)
        return_log_prob: If True, return (p_z, log_p_z) tuple (default: False)
        return_token_log_probs: If True, include per-token log probabilities in the returned tuple (default: False)
        return_token_logits: If True, include raw logits (unnormalised scores before soft-max) for each suffix token in the returned tuple (default: False)

    Returns:
        If no extra flags: p_z only.
        If return_log_prob: (p_z, log_p_z)
        If return_token_log_probs: (p_z, log_p_z, token_log_probs)
        If return_token_logits: (p_z, log_p_z, token_log_probs, token_logits)
        The two boolean flags can be combined; the order of outputs is always
        (p_z, log_p_z, token_log_probs[, token_logits]) where later elements are
        included only if the corresponding flag is True.
    """
    # Tokenize
    prefix_ids = tokenizer.encode(prefix, return_tensors="pt")
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    # Handle empty suffix
    if len(suffix_ids) == 0:
        if return_token_log_probs:
            return (1.0, 0.0, [])
        elif return_log_prob:
            return (1.0, 0.0)
        else:
            return 1.0

    full_ids = torch.cat([prefix_ids, torch.tensor(suffix_ids).unsqueeze(0)], dim=1)

    # Move to same device as model
    device = next(model.parameters()).device
    full_ids = full_ids.to(device)

    # Get logits in one forward pass
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    # Apply temperature
    if temperature != 1.0:
        print(f"Applying temperature scaling with temperature={temperature}", flush=True)
        logits = logits / temperature
    else:
        print(f"No temperature scaling applied", flush=True)

    # Apply top-k filtering
    if top_k is not None:
        print(f"Applying top-k filtering with top_k={top_k}", flush=True)
        v, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < v[:, :, [-1]]] = -float("inf")
    else:
        print(f"No top-k filtering applied", flush=True)

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Sum log probs for suffix tokens
    total_log_prob = 0.0
    token_log_probs = []
    token_logits: List[float] = []
    prefix_len = prefix_ids.shape[1]

    for i in range(len(suffix_ids)):
        token_id = suffix_ids[i]
        time_index = prefix_len + i - 1
        log_prob = log_probs[0, time_index, token_id].item()
        raw_logit = logits[0, time_index, token_id].item()
        total_log_prob += log_prob
        token_log_probs.append(log_prob)
        token_logits.append(raw_logit)

    p_z = torch.exp(torch.tensor(total_log_prob)).item()
    
    # Build return tuple dynamically based on requested information
    if not (return_log_prob or return_token_log_probs or return_token_logits):
        return p_z

    outputs: List[Union[float, List[float]]] = [p_z, total_log_prob]
    if return_token_log_probs:
        outputs.append(token_log_probs)
    if return_token_logits:
        outputs.append(token_logits)

    return tuple(outputs) if len(outputs) > 1 else outputs[0]



def _tokenize_example(tokenizer: transformers.PreTrainedTokenizerBase):
    """Tokenize the concatenation of prefix and suffix without adding special tokens."""
    ids = tokenizer(prefix_singleton + suffix_singleton, add_special_tokens=False)["input_ids"]
    return ids


def _build_levanter_model(prompt_len: int, vocab_size: int) -> LlamaLMHeadModel:
    """Constructs a Levanter Llama-3-8B model with *hard-coded* config values (ignoring HF config).

    The values come from the yaml snippet:
      seq_len: 100
      hidden_dim: 4096
      intermediate_dim: 14336
      num_layers: 32
      num_heads: 32
      num_kv_heads: 8
      flash_attention_block_size: 512
      use_bias: False
      use_layer_norm_weight: True
      initializer_range: 0.02
      rope.type: "llama3"
    """

    seq_len = max(100, prompt_len)  # ensure the sequence fits

    lev_config = LlamaConfig(
        seq_len=seq_len,
        hidden_dim=_HIDDEN_DIM,
        intermediate_dim=_INTERMEDIATE_DIM,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        num_kv_heads=_NUM_KV_HEADS,
        flash_attention_block_size=512,
        use_bias=False,
        use_layer_norm_weight=True,
        initializer_range=0.02,
        rope=Llama3RotaryEmbeddingsConfig(),
        tie_word_embeddings=(LLAMA3_VARIANT == "1B"),
        # disable checkpointing/scan for clarity & determinism in the test
        gradient_checkpointing=False,
        scan_layers=True,
    )

    converter = lev_config.hf_checkpoint_converter().replaced(reference_checkpoint=MODEL_ID)
    model = converter.load_pretrained(
        LlamaLMHeadModel,
        ref=MODEL_ID,
        dtype=jax.numpy.float32,  # match HF precision
        config=lev_config,
        resize_vocab_to_match_tokenizer=False,
    )
    return model


def _collect_levanter_hidden_states(
    model: LlamaLMHeadModel, input_ids: NamedArray, mask: AttentionMask
) -> List[np.ndarray]:
    """Runs the input through each layer, capturing outputs after embeddings and after every layer."""
    hidden: List[np.ndarray] = []

    # Initial embeddings output (input to the first transformer layer)
    x = model.embeddings.embed(input_ids)

    # Determine how to iterate over the per-layer modules inside the Stacked/BlockSeq container.
    container = model.transformer.layers
    # For Stacked we can get a Python list via unstacked()
    if hasattr(container, "unstacked"):
        layers_iter = container.unstacked()
    elif hasattr(container, "layers"):
        layers_iter = container.layers  # BlockSeq path
    elif hasattr(container, "blocks"):
        layers_iter = container.blocks  # fallback
    else:
        # Fallback: assume the container is iterable or indexable
        try:
            layers_iter = list(container)  # type: ignore[arg-type]
        except TypeError:
            # Last resort: index by number of layers
            layers_iter = [container[i] for i in range(model.config.num_layers)]  # type: ignore[index]

    # Mirror HF semantics:
    #   • Append the hidden state *before* each decoder layer.
    #   • After the loop, append the output of the final global RMSNorm.

    for layer in layers_iter:
        # Capture current hidden state (input to this layer)
        hidden.append(np.array(jax.device_get(x.array)))
        # Forward through the layer
        x = layer(x, mask)

    # After all layers, apply the transformer-level RMSNorm and record it.
    x_norm = model.transformer.norm(x)
    hidden.append(np.array(jax.device_get(x_norm.array)))

    # Resulting list layout (matches HF):
    #   index 0            : embeddings (input to layer 0)
    #   index 1..num_layers-1 : inputs to layers 1..(num_layers-1)
    #   index num_layers    : final transformer.norm output
    # Therefore length = num_layers + 1
    return hidden


def _collect_hf_hidden_states(
    model: transformers.PreTrainedModel, input_ids: torch.Tensor
) -> List[np.ndarray]:
    with torch.no_grad():
        out = model(input_ids, use_cache=True, output_hidden_states=True, return_dict=True)
    # HF returns a tuple (embed_out, layer1_out, ... layerN_out)
    return [hs.cpu().float().numpy() for hs in out.hidden_states]


@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")
def test_llama_prefix_intermediates_close():
    """Compare intermediate activations of Levanter vs HuggingFace Llama-3-{LLAMA3_VARIANT} on a fixed prompt."""
    # ------------------------------------------------------------------
    # Tokenize once so both frameworks receive identical input IDs.
    # ------------------------------------------------------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_list = _tokenize_example(tokenizer)
    prompt_len = len(ids_list)

    # Build torch tensor for HF (batch size = 1)
    input_ids_torch = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0)
    input_ids_torch = input_ids_torch.to("cuda")

    # ------------------------------------------------------------------
    # Load HF model (fp16 on single GPU if available, else CPU float32)
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Always load in full float32 precision for an apples-to-apples comparison.
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    hf_model.eval()

    # Assert we are indeed using Llama-3 rotary embeddings on both sides
    assert hf_model.config.rope_scaling is not None and hf_model.config.rope_scaling.get("rope_type") == "llama3", "HF model is not configured with Llama-3 RoPE"

    hf_hidden = _collect_hf_hidden_states(hf_model, input_ids_torch)

    for i, hs in enumerate(hf_hidden):
        print(f"HF layer {i} shape {hs.shape}", flush=True)

    # ------------------------------------------------------------------
    # Prepare Levanter model & JAX input
    # ------------------------------------------------------------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    # Assert Levanter model is using Llama-3 RoPE
    assert isinstance(lev_model.config.rope, Llama3RotaryEmbeddingsConfig), "Levanter model is not using Llama-3 RoPE"

    Batch = Axis("batch", 1)
    Pos = lev_model.config.Pos  # may be >= prompt_len

    # Pad the prompt up to Pos.size so shapes match if needed
    pad_len = Pos.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    input_ids_named = hax.named(input_ids_np, (Batch, Pos))
    causal_mask = AttentionMask.causal()

    lev_hidden = _collect_levanter_hidden_states(lev_model, input_ids_named, causal_mask)

    # Ensure same number of hidden layers
    assert len(lev_hidden) == len(hf_hidden), f"Mismatch in layer count: {len(lev_hidden)} vs {len(hf_hidden)}"

    # ------------------------------------------------------------------
    # Compare layer-wise (slice Levanter tensors to actual prompt length)
    # ------------------------------------------------------------------
    print("HF length", len(hf_hidden), "Lev length", len(lev_hidden), flush=True) 

    for i, (lev, hf) in enumerate(zip(lev_hidden, hf_hidden)):
        # Slice Levanter to first prompt_len positions so shapes line up
        lev_slice = lev[:, :prompt_len, :]
        print(f"Lev layer {i} slice shape {lev_slice.shape}")
        print(f"HF layer {i} shape {hf.shape}")

        assert lev_slice.shape == hf.shape, f"Shape mismatch at layer {i}: {lev_slice.shape} vs {hf.shape}"
        # ------------------------------------------------------------------
        # Detailed numeric comparison. We still attempt the strict chex check,
        # but if it fails we print a human-readable diagnostic that pinpoints
        # *where* the largest discrepancies are in terms of token position and
        # attention head within the 4096-d hidden state.
        # ------------------------------------------------------------------
        rtol = 1e-6
        atol = 1e-6

        try:
            chex.assert_trees_all_close(
                lev_slice.astype(np.float32), hf.astype(np.float32), rtol=rtol, atol=atol
            )
        except AssertionError as e:
            # Compute absolute differences and identify elements outside the tolerance.
            diff = np.abs(lev_slice.astype(np.float32) - hf.astype(np.float32))
            tol = atol + rtol * np.abs(hf.astype(np.float32))
            mismatches = diff > tol

            num_mismatch = int(np.count_nonzero(mismatches))
            total = diff.size
            max_diff = float(diff.max())

            print(
                f"\n❌  Layer {i}: {num_mismatch}/{total} elements exceed tolerance. "
                f"Max |Δ| = {max_diff:.6f}",
                flush=True,
            )

            # Show the top-K largest discrepancies.
            TOP_K = 14
            if num_mismatch > 0:
                mismatch_coords = np.column_stack(np.where(mismatches))
                diff_values = diff[mismatches]
                topk_idx = np.argsort(diff_values)[-TOP_K:][::-1]

                head_dim = lev_model.config.hidden_dim // lev_model.config.num_heads

                print("Rank | Batch | Pos | Hidden | Head | Dim-in-Head |   Lev   |    HF    |  |Δ|", flush=True)
                print("-----|-------|-----|--------|------|-------------|---------|----------|------", flush=True)

                for rank, idx_flat in enumerate(topk_idx, start=1):
                    b, p, d = mismatch_coords[idx_flat]
                    head = d // head_dim
                    dim_in_head = d % head_dim
                    lev_val = float(lev_slice[b, p, d])
                    hf_val = float(hf[b, p, d])
                    delta = float(diff[b, p, d])
                    print(
                        f"{rank:>4} | {b:>5} | {p:>3} | {d:>6} | {head:>4} | {dim_in_head:>11} "
                        f"| {lev_val:+.6f} | {hf_val:+.6f} | {delta:.6f}",
                        flush=True,
                    )

            # Re-raise so that pytest still marks the test as failed.
            raise

# ============================================================================
# New diagnostic test: focuses ONLY on the final decoder block and end-to-end
# log-probabilities for the suffix.  This helps localise the large fp32
# mismatch we observed.
# ============================================================================

@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")
def test_llama_last_block_and_logprobs():
    """Isolate the last block (layer 32) and compare its sub-components as well
    as final log-probabilities of the suffix tokens."""

    import torch.nn.functional as F  # local import to keep global deps unchanged

    # ---------------- Tokenisation ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_list = _tokenize_example(tokenizer)
    prompt_len = len(ids_list)

    # hf tensors
    input_ids_torch = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map={"": device})
    hf_model.eval()

    # Assert we are indeed using Llama-3 rotary embeddings on both sides
    assert hf_model.config.rope_scaling is not None and hf_model.config.rope_scaling.get("rope_type") == "llama3", "HF model is not configured with Llama-3 RoPE"

    with torch.no_grad():
        hf_outputs = hf_model(input_ids_torch, use_cache=True, output_hidden_states=True, return_dict=True)
    hf_hidden_states = [hs.cpu().float().numpy() for hs in hf_outputs.hidden_states]

    # ---------------- Levanter side ----------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    # Assert Levanter model is using Llama-3 RoPE
    assert isinstance(lev_model.config.rope, Llama3RotaryEmbeddingsConfig), "Levanter model is not using Llama-3 RoPE"

    Batch = Axis("batch", 1)
    Pos = lev_model.config.Pos
    pad_len = Pos.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    input_ids_named = hax.named(input_ids_np, (Batch, Pos))
    causal_mask = AttentionMask.causal()

    lev_hidden_states = _collect_levanter_hidden_states(lev_model, input_ids_named, causal_mask)

    # ---------------- Sanity: layers 0..31 must be close ----------------
    chex.assert_trees_all_close(
        lev_hidden_states[-2][:, :prompt_len, :].astype(np.float32),
        hf_hidden_states[-2].astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )

    # ---------------- Step-by-step through the last block ---------------
    # Grab Levanter last block module
    container = lev_model.transformer.layers
    if hasattr(container, "unstacked"):
        lev_last_block = container.unstacked()[-1]
    elif hasattr(container, "layers"):
        lev_last_block = container.layers[-1]
    else:
        lev_last_block = container[-1]

    # Inputs as NamedArrays
    x_prev_lev = lev_hidden_states[-2]  # numpy array
    x_prev_lev_named = hax.named(x_prev_lev, (Batch, Pos, lev_model.config.Embed))

    x_prev_hf = hf_hidden_states[-2]  # numpy array (1, prompt_len, hidden_dim)

    # 1. input-layernorm
    ln1_lev = lev_last_block.input_layernorm(x_prev_lev_named)
    ln1_hf_t = (
        hf_model.model.layers[-1]
        .input_layernorm(torch.from_numpy(x_prev_hf).to(device))
        .detach()
        .cpu()
        .numpy()
    )
    ln1_lev_slice = ln1_lev.array[:, :prompt_len, :]
    max_diff_ln1 = float(np.max(np.abs(ln1_lev_slice.astype(np.float32) - ln1_hf_t.astype(np.float32))))
    print(f"Max |Δ| after input LN: {max_diff_ln1:.6e}", flush=True)
    chex.assert_trees_all_close(ln1_lev_slice.astype(np.float32), ln1_hf_t.astype(np.float32), rtol=1e-3, atol=1e-3)

    # 2. Attention output (no residual yet)
    attn_out_lev_full = lev_last_block.self_attn(x=ln1_lev, mask=causal_mask)
    attn_out_lev = attn_out_lev_full.array[:, :prompt_len, :]

    #   HF attention path — explicit causal mask compatible with LlamaAttention
    seq_len = prompt_len
    causal = torch.full((seq_len, seq_len), float("-inf"), device=device)
    causal = torch.triu(causal, diagonal=1)
    attn_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
    with torch.no_grad():
        ln1_hf_t_torch = torch.from_numpy(ln1_hf_t).to(device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = hf_model.model.rotary_emb(ln1_hf_t_torch, position_ids)
        attn_out_hf_t, _ = hf_model.model.layers[-1].self_attn(
            ln1_hf_t_torch,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
        )
    attn_out_hf = attn_out_hf_t.detach().cpu().numpy()
    max_diff_attn = float(np.max(np.abs(attn_out_lev.astype(np.float32) - attn_out_hf.astype(np.float32))))
    print(f"Max |Δ| after attention: {max_diff_attn:.6e}", flush=True)
    chex.assert_trees_all_close(attn_out_lev.astype(np.float32), attn_out_hf.astype(np.float32), rtol=1e-3, atol=1e-3)

    # 3. Residual add after attention
    x_mid_lev_full = x_prev_lev_named + attn_out_lev_full
    x_mid_lev = x_mid_lev_full.array[:, :prompt_len, :]
    x_mid_hf = x_prev_hf + attn_out_hf

    # 4. post-attention layernorm
    ln2_lev_full = lev_last_block.post_attention_layernorm(hax.named(x_mid_lev_full.array, (Batch, Pos, lev_model.config.Embed)))
    ln2_lev = ln2_lev_full.array[:, :prompt_len, :]
    ln2_hf_t = (
        hf_model.model.layers[-1]
        .post_attention_layernorm(torch.from_numpy(x_mid_hf).to(device))
        .detach()
        .cpu()
        .numpy()
    )
    max_diff_ln2 = float(np.max(np.abs(ln2_lev.astype(np.float32) - ln2_hf_t.astype(np.float32))))
    print(f"Max |Δ| after post-attention LN: {max_diff_ln2:.6e}", flush=True)
    chex.assert_trees_all_close(ln2_lev.astype(np.float32), ln2_hf_t.astype(np.float32), rtol=1e-3, atol=1e-3)

    # 5. MLP output
    mlp_out_lev_full = lev_last_block.mlp(hax.named(ln2_lev_full.array, (Batch, Pos, lev_model.config.Embed)))
    mlp_out_lev = mlp_out_lev_full.array[:, :prompt_len, :]
    mlp_out_hf_t = hf_model.model.layers[-1].mlp(torch.from_numpy(ln2_hf_t).to(device))
    mlp_out_hf = mlp_out_hf_t.detach().cpu().numpy()
    max_diff_mlp = float(np.max(np.abs(mlp_out_lev.astype(np.float32) - mlp_out_hf.astype(np.float32))))
    print(f"Max |Δ| after MLP: {max_diff_mlp:.6e}", flush=True)
    chex.assert_trees_all_close(mlp_out_lev.astype(np.float32), mlp_out_hf.astype(np.float32), rtol=1e-3, atol=1e-3)

    # 6. Final residual add after MLP — this should reproduce full layer output
    last_out_lev = x_mid_lev + mlp_out_lev
    last_out_hf = (x_mid_hf + mlp_out_hf)[:, :prompt_len, :]
    max_diff_last = float(np.max(np.abs(last_out_lev.astype(np.float32) - last_out_hf.astype(np.float32))))
    print(f"Max |Δ| after full last block: {max_diff_last:.6e}", flush=True)
    chex.assert_trees_all_close(last_out_lev.astype(np.float32), last_out_hf.astype(np.float32), rtol=1e-3, atol=1e-3)

    # ---------------- Log-probability comparison ------------------------
    # HF helper returns p_z, log_p_z, token_log_probs, token_logits
    p_z_hf, log_p_z_hf, token_lp_hf = compute_extraction_prob(
        hf_model, tokenizer, prefix_singleton, suffix_singleton, return_log_prob=True, return_token_log_probs=True
    )[:3]

    # Levanter side — mirror the helper logic exactly
    tokens_full = ids_list  # prefix+suffix ids, no padding
    prefix_len = len(prefix_singleton_ids := tokenizer(prefix_singleton, add_special_tokens=False)["input_ids"])

    input_np = np.array([tokens_full], dtype=np.int32)
    Pos_short = Axis("position", len(tokens_full))
    lev_input_named = hax.named(input_np, (Batch, Pos_short))
    lev_logits_named = lev_model(lev_input_named, attn_mask=AttentionMask.causal())  # (1, P, Vocab)
    lev_logits = lev_logits_named.array
    lev_lp_named = log_softmax(lev_logits_named, axis=lev_model.Vocab)
    lev_lp = jax.nn.log_softmax(lev_logits, axis=-1)

    # compare results from named tensor to using jax.nn.log_softmax
    chex.assert_trees_all_close(lev_lp_named.array, lev_lp, rtol=1e-4, atol=1e-4)

    suffix_ids = tokenizer(suffix_singleton, add_special_tokens=False)["input_ids"]
    suffix_len = len(suffix_ids)

    lev_token_lp = []
    for i, tok in enumerate(suffix_ids):
        time_idx = prefix_len + i - 1  # same as HF helper
        lp_val = float(lev_lp[0, time_idx, tok])
        lev_token_lp.append(lp_val)

    log_p_z_lev = float(sum(lev_token_lp))
    p_z_lev = float(np.exp(log_p_z_lev))

    print(f"HF log_p_z: {log_p_z_hf:.6f}  Lev log_p_z: {log_p_z_lev:.6f}", flush=True)

    # --- Logits divergence diagnostic ---
    hf_logits = hf_outputs.logits.cpu().float().numpy()
    # slice Levanter logits to same seq_len (no padding)
    lev_logits_np = lev_logits[:, :hf_logits.shape[1], :]
    max_diff_logits = float(np.max(np.abs(lev_logits_np.astype(np.float32) - hf_logits.astype(np.float32))))
    print(f"Max |Δ| in raw logits over full sequence: {max_diff_logits:.6e}", flush=True)


    # ---------------- Per-token diagnostic table -------------------
    hf_logits_np = hf_outputs.logits.cpu().float().numpy()
    hf_log_probs_np = F.log_softmax(torch.from_numpy(hf_logits_np), dim=-1).numpy()

    print("\nToken | ID | String | Lev Logit | Lev LogP | Lev P |  HF Logit |  HF LogP |  HF P | |Δ LogP|", flush=True)
    print("----- |----|--------|-----------|----------|-------|-----------|----------|-------|---------", flush=True)

    for i, tok in enumerate(suffix_ids):
        time_idx = prefix_len + i - 1
        tok_str = tokenizer.decode([tok]).replace("\n", "\\n")

        lev_logit = float(lev_logits[0, time_idx, tok])
        lev_logp = float(lev_lp[0, time_idx, tok])
        lev_p = float(np.exp(lev_logp))

        hf_logit = float(hf_logits_np[0, time_idx, tok])
        hf_logp = float(hf_log_probs_np[0, time_idx, tok])
        hf_p = float(np.exp(hf_logp))

        delta_logp = lev_logp - hf_logp

        print(
            f"{i:>4} | {tok:>5} | {tok_str} | {lev_logit:+.4f} | {lev_logp:+.5f} | {lev_p:.5f} "
            f"| {hf_logit:+.4f} | {hf_logp:+.5f} | {hf_p:.5f} | {delta_logp:+.2e}",
            flush=True,
        )

    # Compare per-token log-probs as well (HF helper gives suffix-only tokens)
    np.testing.assert_allclose(np.array(lev_token_lp, dtype=np.float32), np.array(token_lp_hf, dtype=np.float32), rtol=1e-3, atol=1e-3)

    # Compare total log-probability
    np.testing.assert_allclose(log_p_z_lev, log_p_z_hf, rtol=1e-3, atol=1e-3)


@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")

def test_final_activation_comparison():
    """Compare the final hidden activations (after the last layer norm, before the LM head)
    of Levanter vs HuggingFace Llama-3-{LLAMA3_VARIANT} on the same prompt.
    """

    # ---------------- Tokenisation ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_list = _tokenize_example(tokenizer)
    prompt_len = len(ids_list)

    # Hugging Face tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids_torch = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to(device)

    # ---------------- Load HF model ----------------
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    hf_model.eval()

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids_torch, use_cache=True, output_hidden_states=True, return_dict=True
        )
    hf_final_activations = hf_outputs.hidden_states[-1].cpu().float().numpy()  # (1, P, E)

    # ---------------- Levanter side ----------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    Batch = Axis("batch", 1)
    Pos = lev_model.config.Pos  # may be >= prompt_len

    pad_len = Pos.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    lev_input_named = hax.named(input_ids_np, (Batch, Pos))
    causal_mask = AttentionMask.causal()

    lev_activations_named = lev_model.activations(lev_input_named, attn_mask=causal_mask)
    lev_activations = lev_activations_named.array[:, :prompt_len, :]  # slice to prompt length

    # ---------------- Comparison ----------------
    assert lev_activations.shape == hf_final_activations.shape, (
        f"Shape mismatch: {lev_activations.shape} vs {hf_final_activations.shape}"
    )

    # Use stricter tolerances here, and emit diagnostics on failure
    rtol = 1e-4
    atol = 1e-4

    try:
        chex.assert_trees_all_close(
            lev_activations.astype(np.float32),
            hf_final_activations.astype(np.float32),
            rtol=rtol,
            atol=atol,
        )
    except AssertionError:
        diff = np.abs(lev_activations.astype(np.float32) - hf_final_activations.astype(np.float32))
        tol = atol + rtol * np.abs(hf_final_activations.astype(np.float32))
        mismatches = diff > tol

        num_mismatch = int(np.count_nonzero(mismatches))
        total = diff.size
        max_diff = float(diff.max())

        print(
            f"\n❌  Final activations: {num_mismatch}/{total} elements exceed tolerance. Max |Δ| = {max_diff:.6f}",
            flush=True,
        )

        if num_mismatch > 0:
            mismatch_coords = np.column_stack(np.where(mismatches))
            diff_values = diff[mismatches]
            TOP_K = 20
            topk_idx = np.argsort(diff_values)[-TOP_K:][::-1]

            head_dim = lev_model.config.hidden_dim // lev_model.config.num_heads

            print("Rank | Batch | Pos | Hidden | Head | Dim-in-Head |   Lev   |    HF    |  |Δ|", flush=True)
            print("-----|-------|-----|--------|------|-------------|---------|----------|------", flush=True)

            for rank, idx_flat in enumerate(topk_idx, start=1):
                b, p, d = mismatch_coords[idx_flat]
                head = d // head_dim
                dim_in_head = d % head_dim
                lev_val = float(lev_activations[b, p, d])
                hf_val = float(hf_final_activations[b, p, d])
                delta = float(diff[b, p, d])
                print(
                    f"{rank:>4} | {b:>5} | {p:>3} | {d:>6} | {head:>4} | {dim_in_head:>11} "
                    f"| {lev_val:+.6f} | {hf_val:+.6f} | {delta:.6f}",
                    flush=True,
                )

        # Re-raise to ensure the test still fails if tolerance exceeded
        raise


@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")

def test_rmsnorm_equivalence_detailed():
    """Diagnostic test that compares the final RMSNorm layer between HuggingFace and Levanter.

    It checks:
      1. ε (variance_epsilon) values.
      2. Weight vectors.
      3. Functional behaviour on identical pre-norm activations.
      4. End-to-end equality of post-norm activations.
    Prints detailed stats so discrepancies can be pinpointed easily.
    """

    # ---------------- Tokenisation ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_list = _tokenize_example(tokenizer)
    prompt_len = len(ids_list)

    # ---------------- HuggingFace model ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    hf_model.eval()

    input_ids_torch = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hf_outputs = hf_model(
            input_ids_torch, use_cache=False, output_hidden_states=True, return_dict=True
        )
    hf_hidden = [hs.cpu().float().numpy() for hs in hf_outputs.hidden_states]

    pre_norm_hf = hf_hidden[-2]  # before global RMSNorm
    post_norm_hf = hf_hidden[-1]  # after global RMSNorm

    # ---------------- Levanter model ----------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    Batch = Axis("batch", 1)
    Pos_full = lev_model.config.Pos
    pad_len = Pos_full.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    lev_input_named = hax.named(input_ids_np, (Batch, Pos_full))
    causal_mask = AttentionMask.causal()

    lev_hidden = _collect_levanter_hidden_states(lev_model, lev_input_named, causal_mask)
    pre_norm_lev = lev_hidden[-2][:, :prompt_len, :]
    post_norm_lev = lev_hidden[-1][:, :prompt_len, :]

    # ---------------- Config / parameter checks ----------------
    hf_eps = float(hf_model.model.norm.variance_epsilon)
    lev_eps = float(lev_model.transformer.norm.eps)
    print(f"ε  HF={hf_eps}   Levanter={lev_eps}")

    hf_w = hf_model.model.norm.weight.detach().cpu().numpy()
    lev_w = lev_model.transformer.norm.weight.array
    print("RMSNorm weight max |Δ|:", float(np.max(np.abs(hf_w - lev_w))))
    print("RMSNorm weight mean |Δ|:", float(np.mean(np.abs(hf_w - lev_w))))

    # ---------------- Functional comparison on identical input ----------------
    # Pass the same pre-norm activations through each implementation
    with torch.no_grad():
        out_hf_ident = hf_model.model.norm(torch.from_numpy(pre_norm_hf).to(device)).cpu().numpy()

    Pos_short = Axis("position", prompt_len)
    x_named = hax.named(pre_norm_hf.astype(np.float32), (Batch, Pos_short, lev_model.config.Embed))
    out_lev_ident = lev_model.transformer.norm(x_named).array

    diff_ident = np.max(np.abs(out_hf_ident - out_lev_ident))
    print("Functional diff (identical input) max |Δ|:", float(diff_ident))

    # ---------------- Assertions ----------------
    chex.assert_trees_all_close(
        out_hf_ident.astype(np.float32), out_lev_ident.astype(np.float32), rtol=1e-4, atol=1e-4
    )
    chex.assert_trees_all_close(
        post_norm_hf.astype(np.float32), post_norm_lev.astype(np.float32), rtol=1e-4, atol=1e-4
    )

@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")
def test_llama_prefix_first_layer():
    """Detailed component-wise comparison of the *first* transformer layer (decoder block 0)
    between Levanter and HuggingFace.  Shows where divergence originates if any.
    """
    RTOL = 1e-5
    ATOL = 1e-5

    import torch.nn.functional as F  # local to keep global deps minimal

    # ---------------- Tokenisation ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_list = _tokenize_example(tokenizer)
    prompt_len = len(ids_list)

    # ---------------- Hugging Face side ----------------
    input_ids_torch = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids_torch = input_ids_torch.to(device)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    hf_model.eval()

    # Ensure HF model really uses Llama-3 rotary embeddings
    assert hf_model.config.rope_scaling is not None and hf_model.config.rope_scaling.get("rope_type") == "llama3", "HF model is not configured with Llama-3 RoPE"

    with torch.no_grad():
        hf_outputs = hf_model(input_ids_torch, use_cache=True, output_hidden_states=True, return_dict=True)
    hf_hidden_states = [hs.cpu().float().numpy() for hs in hf_outputs.hidden_states]

    # ---------------- Levanter side ----------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    # Assert Levanter model is using Llama-3 RoPE
    assert isinstance(lev_model.config.rope, Llama3RotaryEmbeddingsConfig), "Levanter model is not using Llama-3 RoPE"

    Batch = Axis("batch", 1)
    Pos_full = lev_model.config.Pos  # may be >= prompt_len
    pad_len = Pos_full.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    lev_input_named = hax.named(input_ids_np, (Batch, Pos_full))
    causal_mask = AttentionMask.causal()

    lev_hidden_states = _collect_levanter_hidden_states(lev_model, lev_input_named, causal_mask)

    # ---------------- Sanity: embedding outputs must be close ----------------
    chex.assert_trees_all_close(
        lev_hidden_states[0][:, :prompt_len, :].astype(np.float32),
        hf_hidden_states[0].astype(np.float32),
        rtol=RTOL,
        atol=ATOL,
    )

    # ---------------- Step-by-step through the FIRST block ----------------
    # Grab Levanter first block module
    container = lev_model.transformer.layers
    if hasattr(container, "unstacked"):
        lev_first_block = container.unstacked()[0]
    elif hasattr(container, "layers"):
        lev_first_block = container.layers[0]
    else:
        lev_first_block = container[0]

    # Inputs as NamedArrays
    x_prev_lev = lev_hidden_states[0]  # numpy array after embeddings
    x_prev_lev_named = hax.named(x_prev_lev, (Batch, Pos_full, lev_model.config.Embed))

    x_prev_hf = hf_hidden_states[0]  # numpy array (1, prompt_len, hidden_dim)

    # 1. input-layernorm
    ln1_lev = lev_first_block.input_layernorm(x_prev_lev_named)
    ln1_hf_t = (
        hf_model.model.layers[0]
        .input_layernorm(torch.from_numpy(x_prev_hf).to(device))
        .detach()
        .cpu()
        .numpy()
    )
    ln1_lev_slice = ln1_lev.array[:, :prompt_len, :]
    max_diff_ln1 = float(np.max(np.abs(ln1_lev_slice.astype(np.float32) - ln1_hf_t.astype(np.float32))))
    print(f"Max |Δ| after input LN: {max_diff_ln1:.6e}")
    chex.assert_trees_all_close(ln1_lev_slice.astype(np.float32), ln1_hf_t.astype(np.float32), rtol=RTOL, atol=ATOL)

    # 2. Attention output (no residual yet)
    attn_out_lev_full = lev_first_block.self_attn(x=ln1_lev, mask=causal_mask)
    attn_out_lev = attn_out_lev_full.array[:, :prompt_len, :]

    #   HF attention path — explicit causal mask compatible with LlamaAttention
    seq_len = prompt_len
    causal = torch.full((seq_len, seq_len), float("-inf"), device=device)
    causal = torch.triu(causal, diagonal=1)
    attn_mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, src_len)
    with torch.no_grad():
        ln1_hf_t_torch = torch.from_numpy(ln1_hf_t).to(device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = hf_model.model.rotary_emb(ln1_hf_t_torch, position_ids)
        attn_out_hf_t, _ = hf_model.model.layers[0].self_attn(
            ln1_hf_t_torch,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
        )
    attn_out_hf = attn_out_hf_t.detach().cpu().numpy()
    max_diff_attn = float(np.max(np.abs(attn_out_lev.astype(np.float32) - attn_out_hf.astype(np.float32))))
    print(f"Max |Δ| after attention: {max_diff_attn:.6e}")
    chex.assert_trees_all_close(attn_out_lev.astype(np.float32), attn_out_hf.astype(np.float32), rtol=RTOL, atol=ATOL)

    # 3. Residual add after attention
    x_mid_lev_full = x_prev_lev_named + attn_out_lev_full
    x_mid_lev = x_mid_lev_full.array[:, :prompt_len, :]
    x_mid_hf = x_prev_hf + attn_out_hf

    # 4. post-attention layernorm
    ln2_lev_full = lev_first_block.post_attention_layernorm(hax.named(x_mid_lev_full.array, (Batch, Pos_full, lev_model.config.Embed)))
    ln2_lev = ln2_lev_full.array[:, :prompt_len, :]
    ln2_hf_t = (
        hf_model.model.layers[0]
        .post_attention_layernorm(torch.from_numpy(x_mid_hf).to(device))
        .detach()
        .cpu()
        .numpy()
    )
    max_diff_ln2 = float(np.max(np.abs(ln2_lev.astype(np.float32) - ln2_hf_t.astype(np.float32))))
    print(f"Max |Δ| after post-attention LN: {max_diff_ln2:.6e}")
    chex.assert_trees_all_close(ln2_lev.astype(np.float32), ln2_hf_t.astype(np.float32), rtol=RTOL, atol=ATOL)

    # 5. MLP output
    mlp_out_lev_full = lev_first_block.mlp(hax.named(ln2_lev_full.array, (Batch, Pos_full, lev_model.config.Embed)))
    mlp_out_lev = mlp_out_lev_full.array[:, :prompt_len, :]
    mlp_out_hf_t = hf_model.model.layers[0].mlp(torch.from_numpy(ln2_hf_t).to(device))
    mlp_out_hf = mlp_out_hf_t.detach().cpu().numpy()
    max_diff_mlp = float(np.max(np.abs(mlp_out_lev.astype(np.float32) - mlp_out_hf.astype(np.float32))))
    print(f"Max |Δ| after MLP: {max_diff_mlp:.6e}")
    chex.assert_trees_all_close(mlp_out_lev.astype(np.float32), mlp_out_hf.astype(np.float32), rtol=RTOL, atol=ATOL)

    # 6. Final residual add after MLP — reproduces layer output
    last_out_lev = x_mid_lev + mlp_out_lev
    last_out_hf = (x_mid_hf + mlp_out_hf)[:, :prompt_len, :]
    max_diff_last = float(np.max(np.abs(last_out_lev.astype(np.float32) - last_out_hf.astype(np.float32))))
    print(f"Max |Δ| after full first block: {max_diff_last:.6e}")
    chex.assert_trees_all_close(last_out_lev.astype(np.float32), last_out_hf.astype(np.float32), rtol=RTOL, atol=ATOL)

@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")
def test_llama_prefix_first_layer_no_attention():
    """Like `test_llama_prefix_first_layer` but forces **identical** attention outputs (taken from
    the Levanter implementation) into both model paths.  This helps determine if the initial
    divergence originates exclusively from attention or if subsequent sub-layers (LN/MLP) also
    contribute.
    """

    RTOL = 1e-5
    ATOL = 1e-5

    # ---------------- Tokenisation ----------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids_list = _tokenize_example(tokenizer)
    prompt_len = len(ids_list)

    # ---------------- Hugging Face side ----------------
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids_torch = torch.tensor(ids_list, dtype=torch.long).unsqueeze(0).to(device)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    hf_model.eval()

    # ---------------- Levanter side ----------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    # Basic config sanity checks
    assert isinstance(lev_model.config.rope, Llama3RotaryEmbeddingsConfig)

    Batch = Axis("batch", 1)
    Pos_full = lev_model.config.Pos
    pad_len = Pos_full.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    lev_input_named = hax.named(input_ids_np, (Batch, Pos_full))
    causal_mask = AttentionMask.causal()

    # Run to capture hidden states up through embeddings for each model
    with torch.no_grad():
        hf_outputs = hf_model(input_ids_torch, use_cache=True, output_hidden_states=True, return_dict=True)
    hf_hidden_states = [hs.cpu().float().numpy() for hs in hf_outputs.hidden_states]

    lev_hidden_states = _collect_levanter_hidden_states(lev_model, lev_input_named, causal_mask)

    # ---------------- Sanity check embeddings ----------------
    chex.assert_trees_all_close(
        lev_hidden_states[0][:, :prompt_len, :].astype(np.float32),
        hf_hidden_states[0].astype(np.float32),
        rtol=RTOL,
        atol=ATOL,
    )

    # ---------------- Grab first decoder block modules ----------------
    container = lev_model.transformer.layers
    if hasattr(container, "unstacked"):
        lev_first_block = container.unstacked()[0]
    elif hasattr(container, "layers"):
        lev_first_block = container.layers[0]
    else:
        lev_first_block = container[0]

    # Corresponding HF layer
    hf_first_block = hf_model.model.layers[0]

    # Prepare common inputs
    x_prev_lev = lev_hidden_states[0]  # numpy
    x_prev_hf = hf_hidden_states[0]    # numpy

    x_prev_lev_named = hax.named(x_prev_lev, (Batch, Pos_full, lev_model.config.Embed))

    # ---------------- 1. Input LayerNorm ----------------
    ln1_lev = lev_first_block.input_layernorm(x_prev_lev_named)
    ln1_lev_np = ln1_lev.array[:, :prompt_len, :]

    with torch.no_grad():
        ln1_hf_t = hf_first_block.input_layernorm(torch.from_numpy(x_prev_hf).to(device)).cpu().numpy()
    chex.assert_trees_all_close(ln1_lev_np.astype(np.float32), ln1_hf_t.astype(np.float32), rtol=RTOL, atol=ATOL)

    # ---------------- 2. Self-attention (Levanter only) ----------------
    attn_out_lev_full = lev_first_block.self_attn(x=ln1_lev, mask=causal_mask)
    attn_out_lev_np = attn_out_lev_full.array[:, :prompt_len, :]
    print("Used Levanter attention output for both models. Max |attn| value: ", float(np.max(np.abs(attn_out_lev_np))))

    # ---------------- 3. Residual add after attention ----------------
    x_mid_lev_full = x_prev_lev_named + attn_out_lev_full
    x_mid_lev_np = x_mid_lev_full.array[:, :prompt_len, :]

    x_mid_hf_np = x_prev_hf + attn_out_lev_np  # force-identical attention contribution

    # ---------------- 4. Post-attention LayerNorm ----------------
    ln2_lev_full = lev_first_block.post_attention_layernorm(hax.named(x_mid_lev_full.array, (Batch, Pos_full, lev_model.config.Embed)))
    ln2_lev_np = ln2_lev_full.array[:, :prompt_len, :]

    with torch.no_grad():
        ln2_hf_t = hf_first_block.post_attention_layernorm(torch.from_numpy(x_mid_hf_np).to(device)).cpu().numpy()
    chex.assert_trees_all_close(ln2_lev_np.astype(np.float32), ln2_hf_t.astype(np.float32), rtol=RTOL, atol=ATOL)

    # ---------------- 5. MLP ----------------
    mlp_out_lev_full = lev_first_block.mlp(hax.named(ln2_lev_full.array, (Batch, Pos_full, lev_model.config.Embed)))
    mlp_out_lev_np = mlp_out_lev_full.array[:, :prompt_len, :]

    with torch.no_grad():
        mlp_out_hf_t = hf_first_block.mlp(torch.from_numpy(ln2_hf_t).to(device)).cpu().numpy()
    chex.assert_trees_all_close(mlp_out_lev_np.astype(np.float32), mlp_out_hf_t.astype(np.float32), rtol=RTOL, atol=ATOL)

    # ---------------- 6. Final residual add ----------------
    last_out_lev_np = x_mid_lev_np + mlp_out_lev_np
    last_out_hf_np = x_mid_hf_np + mlp_out_hf_t

    max_diff_last = float(np.max(np.abs(last_out_lev_np.astype(np.float32) - last_out_hf_np.astype(np.float32))))
    print(f"Max |Δ| after full first block (shared attention): {max_diff_last:.6e}")
    chex.assert_trees_all_close(last_out_lev_np.astype(np.float32), last_out_hf_np.astype(np.float32), rtol=RTOL, atol=ATOL)

    