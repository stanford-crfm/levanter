import numpy as np
import torch
from jax import random

# src/transformers/models/llama/modeling_llama.py
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

from levanter.models.llama import LlamaRotaryEmbedding


def test_llama_rotary_embedding():
    """Match against HuggingFace's implementation of LlamaRotaryEmbedding."""
    dim = 2048
    seq_len = 2048
    key = random.PRNGKey(0)
    x = random.normal(key, (1, seq_len))
    levanter_rotary_emb = LlamaRotaryEmbedding(dim=dim)
    levanter_output = levanter_rotary_emb(x, seq_len=seq_len)

    hf_rotary_emb = HFLlamaRotaryEmbedding(dim=dim, device="cpu")
    x_torch = torch.from_numpy(np.array(x))
    hf_output = hf_rotary_emb(x_torch, seq_len=seq_len)

    for jax_out, torch_out in zip(levanter_output, hf_output):
        torch_out = torch_out.numpy()
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"
