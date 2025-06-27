import dataclasses
import os
from functools import reduce
from typing import List, Sequence

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

# Additional rotary config import for llama-3 behaviour
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig

# -----------------------------------------------------------------------------
# Hard-coded prompt pieces. These are the same strings used in eval_sliding_lm.py
# -----------------------------------------------------------------------------
prefix_singleton = "They were careless people, Tom and Daisy – they smashed up"
suffix_singleton = " things and creatures and then retreated"

MODEL_ID = "meta-llama/Llama-3.1-8B"


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
        hidden_dim=4096,
        intermediate_dim=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        flash_attention_block_size=512,
        use_bias=False,
        use_layer_norm_weight=True,
        initializer_range=0.02,
        rope=Llama3RotaryEmbeddingsConfig(),
        # disable checkpointing/scan for clarity & determinism in the test
        gradient_checkpointing=False,
        scan_layers=True,
    )

    converter = lev_config.hf_checkpoint_converter().replaced(reference_checkpoint=MODEL_ID)

    model = converter.load_pretrained(
        LlamaLMHeadModel,
        ref=MODEL_ID,
        dtype=jax.numpy.float16,
        config=lev_config,
        resize_vocab_to_match_tokenizer=False,
    )
    return model


def _collect_levanter_hidden_states(
    model: LlamaLMHeadModel, input_ids: NamedArray, mask: AttentionMask
) -> List[np.ndarray]:
    """Runs the input through each layer, capturing outputs after embeddings and after every layer."""
    hidden: List[np.ndarray] = []

    # Embeddings output
    x = model.embeddings.embed(input_ids)
    hidden.append(np.array(jax.device_get(x.array)))  # shape (B, P, E)

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

    # Run through layers sequentially, recording the output of each full layer (after residual add)
    for layer in layers_iter:
        x = layer(x, mask)
        hidden.append(np.array(jax.device_get(x.array)))

    return hidden  # length = num_layers + 1 (embeddings + every layer)


def _collect_hf_hidden_states(
    model: transformers.PreTrainedModel, input_ids: torch.Tensor
) -> List[np.ndarray]:
    with torch.no_grad():
        out = model(input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
    # HF returns a tuple (embed_out, layer1_out, ... layerN_out)
    return [hs.cpu().float().numpy() for hs in out.hidden_states]


@skip_if_no_torch
@skip_if_hf_model_not_accessible(MODEL_ID)
@skip_in_ci("Large 8B model – skipped in CI.")
def test_llama_prefix_intermediates_close():
    """Compare intermediate activations of Levanter vs HuggingFace Llama-3-8B on a fixed prompt."""
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
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map={"": device}
    )
    hf_model.eval()

    hf_hidden = _collect_hf_hidden_states(hf_model, input_ids_torch)

    # ------------------------------------------------------------------
    # Prepare Levanter model & JAX input
    # ------------------------------------------------------------------
    vocab_size = hf_model.config.vocab_size
    lev_model = _build_levanter_model(prompt_len, vocab_size)

    Batch = Axis("batch", 1)
    Pos = lev_model.config.Pos  # may be >= prompt_len

    # Pad the prompt up to Pos.size so shapes match if needed
    pad_len = Pos.size - prompt_len
    ids_padded = ids_list + [tokenizer.pad_token_id] * pad_len
    input_ids_np = np.array([ids_padded], dtype=np.int32)
    input_ids_named = hax.named(input_ids_np, (Batch, Pos))
    causal_mask = AttentionMask.causal()

    lev_hidden = _collect_levanter_hidden_states(lev_model, input_ids_named, causal_mask)

    # Ensure same number of checkpoints
    assert len(lev_hidden) == len(hf_hidden), f"Mismatch in hidden-state count: {len(lev_hidden)} vs {len(hf_hidden)}"

    # ------------------------------------------------------------------
    # Compare layer-wise
    # ------------------------------------------------------------------
    for i, (lev, hf) in enumerate(zip(lev_hidden, hf_hidden)):
        # shapes: (1, P, E). Ensure same ordering
        assert lev.shape == hf.shape, f"Shape mismatch at layer {i}: {lev.shape} vs {hf.shape}"
        chex.assert_trees_all_close(lev.astype(np.float32), hf.astype(np.float32), rtol=1e-4, atol=1e-4) 