import os
from typing import Dict, Any

import numpy as np
import torch
import transformers

import jax
import jax.numpy as jnp

from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

from haliax.state_dict import to_torch_compatible_state_dict

# Import test utilities directly (no cross-test variant logic needed)
from tests.test_utils import (
    skip_if_no_torch,
    skip_if_hf_model_not_accessible,
    skip_in_ci,
)

# ----------------------------------------------------------------------------
# Config – hard-coded values for the Llama-3 1B variant
# ----------------------------------------------------------------------------
LLAMA3_VARIANT = "1B"

_VARIANT_SPEC = {
    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    "hidden_dim": 2048,
    "intermediate_dim": 8192,
    "num_layers": 16,
    "num_heads": 32,
    "num_kv_heads": 8,
}

MODEL_ID = _VARIANT_SPEC["model_id"]

_HIDDEN_DIM = _VARIANT_SPEC["hidden_dim"]
_INTERMEDIATE_DIM = _VARIANT_SPEC["intermediate_dim"]
_NUM_LAYERS = _VARIANT_SPEC["num_layers"]
_NUM_HEADS = _VARIANT_SPEC["num_heads"]
_NUM_KV_HEADS = _VARIANT_SPEC["num_kv_heads"]

print(f"Using model: {MODEL_ID}", flush=True)


# ----------------------------------------------------------------------------
# Levanter model builder (replicated here to avoid cross-test imports)
# ----------------------------------------------------------------------------

def _build_levanter_model(prompt_len: int, vocab_size: int) -> LlamaLMHeadModel:
    """Construct a Levanter Llama-3-1B model matching the HF checkpoint."""

    seq_len = max(100, prompt_len)

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
        tie_word_embeddings=True,
        # Disable checkpointing/scan for determinism in the test
        gradient_checkpointing=False,
        scan_layers=True,
    )

    converter = lev_config.hf_checkpoint_converter().replaced(reference_checkpoint=MODEL_ID)
    model = converter.load_pretrained(
        LlamaLMHeadModel,
        ref=MODEL_ID,
        dtype=jnp.float32,  # match HF precision
        config=lev_config,
        resize_vocab_to_match_tokenizer=False,
    )
    return model


# ----------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------

def _state_dict_to_numpy(sd: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert a Torch state-dict (tensors) -> dict[str, np.ndarray] (CPU, float32)."""
    out: Dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().float().numpy()
        else:
            # buffers such as None or scalars; skip
            continue
    return out


# ----------------------------------------------------------------------------
# Main test
# ----------------------------------------------------------------------------


@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci(f"Large {LLAMA3_VARIANT} model – skipped in CI.")
def test_loading_precision():
    """Ensure Levanter-loaded weights match the original HuggingFace checkpoint to <=1e-6."""
    
    print(f"Starting precision test for {LLAMA3_VARIANT} model", flush=True)

    # ---------------- Load HuggingFace model ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    print(f"Loading HuggingFace model: {MODEL_ID}", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map={"": device}
    )
    hf_model.eval()
    print(f"HuggingFace model loaded successfully", flush=True)

    # Convert HF state-dict to NumPy
    print("Converting HF state dict to NumPy", flush=True)
    for k, v in hf_model.state_dict().items():
        print(f" key is {k}: {v.shape}", flush=True)
    hf_sd_np = _state_dict_to_numpy(hf_model.state_dict())
    hf_keys = set(hf_sd_np.keys())
    print(f"HF model has {len(hf_keys)} parameters", flush=True)

    # ---------------- Build Levanter model ------------------
    # prompt_len=1 is sufficient (seq_len will default to 100)
    print("Building Levanter model", flush=True)
    lev_model = _build_levanter_model(prompt_len=1, vocab_size=hf_model.config.vocab_size)
    print("Levanter model built successfully", flush=True)

    print("Converting Levanter state dict", flush=True)
    lev_sd_np = to_torch_compatible_state_dict(lev_model, flatten=True)
    lev_keys = set(lev_sd_np.keys())

    # Handle tied word embeddings: HuggingFace still stores a separate "lm_head.weight" that is
    # simply a view of the token-embedding matrix.  When `tie_word_embeddings=True`, Levanter
    # exports *only* the embedding weight.  Create an alias so key-sets line up and the tensor
    # comparison still happens.
    if "lm_head.weight" in hf_keys and "lm_head.weight" not in lev_keys:
        lev_sd_np["lm_head.weight"] = lev_sd_np["model.embed_tokens.weight"]
        lev_keys.add("lm_head.weight")
    print(f"Levanter model has {len(lev_keys)} parameters", flush=True)

    # ---------------- Key set sanity ------------------------
    print("Checking key set compatibility", flush=True)
    missing_in_lev = hf_keys - lev_keys
    missing_in_hf = lev_keys - hf_keys

    if missing_in_lev:
        print(f"Keys missing in Levanter: {sorted(missing_in_lev)[:10]}...", flush=True)
    if missing_in_hf:
        print(f"Extra keys in Levanter: {sorted(missing_in_hf)[:10]}...", flush=True)

    assert not missing_in_lev, f"Missing keys in Levanter state_dict: {sorted(missing_in_lev)[:20]}..."
    assert not missing_in_hf, f"Extra keys in Levanter state_dict: {sorted(missing_in_hf)[:20]}..."

    print("Key sets match perfectly", flush=True)

    # ---------------- Value comparisons ---------------------
    atol = 1e-7
    rtol = 0.0  # absolute tolerance only
    
    print(f"Starting value comparisons with atol={atol}, rtol={rtol}", flush=True)

    mismatched_keys = []
    max_diffs = []
    
    for i, k in enumerate(sorted(hf_keys)):
        if i % 50 == 0:  # Print progress every 50 keys
            print(f"Comparing parameter {i+1}/{len(hf_keys)}: {k}", flush=True)
            
        lev_arr = lev_sd_np[k]
        hf_arr = hf_sd_np[k]
        
        assert lev_arr.shape == hf_arr.shape, f"Shape mismatch for {k}: {lev_arr.shape} vs {hf_arr.shape}"
        
        # Calculate max difference for this parameter
        max_diff = np.max(np.abs(lev_arr - hf_arr))
        max_diffs.append(max_diff)
        
        try:
            np.testing.assert_allclose(
                lev_arr, hf_arr, rtol=rtol, atol=atol, err_msg=f"Weight mismatch at key {k} (atol={atol})"
            )
        except AssertionError as e:
            print(f"Mismatch in {k}: max_diff={max_diff:.2e}", flush=True)
            mismatched_keys.append(k)
            raise

    overall_max_diff = np.max(max_diffs)
    print(f"All parameters match within tolerance!", flush=True)
    print(f"Overall maximum difference: {overall_max_diff:.2e}", flush=True)
    print(f"Mean maximum difference: {np.mean(max_diffs):.2e}", flush=True)
    print(f"Test completed successfully for {LLAMA3_VARIANT} model", flush=True) 