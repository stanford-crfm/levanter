import numpy as np
import torch
from jax import random

# src/transformers/models/llama/modeling_llama.py
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

try:
    from transformers.models.llama.modeling_llama import (
        LlamaLinearScalingRotaryEmbedding as HFLlamaLinearScalingRotaryEmbedding,
    )
except ImportError:
    HFLlamaLinearScalingRotaryEmbedding = None
from levanter.models.llama import LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding


def test_llama_rotary_embedding():
    """Match against HuggingFace's implementation of LlamaRotaryEmbedding."""
    dim = 2048
    seq_len = 2048
    key = random.PRNGKey(0)
    device = "cpu"

    def test_levanter_against_hf(levanter_class, hf_class):
        x = random.normal(key, (1, seq_len))
        x_torch = torch.from_numpy(np.array(x))

        levanter_output = levanter_class(x, seq_len=seq_len)
        hf_output = hf_class(x_torch, seq_len=seq_len)

        for jax_out, torch_out in zip(levanter_output, hf_output):
            torch_out = torch_out.numpy()
            assert np.isclose(torch_out, np.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"

    # test LlamaRotaryEmbedding
    test_levanter_against_hf(
        levanter_class=LlamaRotaryEmbedding(dim=dim),
        hf_class=HFLlamaRotaryEmbedding(dim=dim, device=device),
    )

    # test LlamaLinearScalingRotaryEmbedding
    if HFLlamaLinearScalingRotaryEmbedding is not None:
        scaling_factor = 2.0
        test_levanter_against_hf(
            levanter_class=LlamaLinearScalingRotaryEmbedding(dim=dim, scaling_factor=scaling_factor),
            hf_class=HFLlamaLinearScalingRotaryEmbedding(dim=dim, scaling_factor=scaling_factor, device=device),
        )
