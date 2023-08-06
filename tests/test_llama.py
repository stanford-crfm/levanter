import numpy as np
import torch
from jax import random

# The latter 2 classes are only available in HuggingFace's transformers 4.30.0 or later
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding as HFLlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding as HFLlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding as HFLlamaDynamicNTKScalingRotaryEmbedding,
    LlamaAttention as HFLlamaAttention,
)
from levanter.models.llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaAttention,
)


"""
def test_llama_rotary_embedding():
    llama_config = _get_llama_config()
    hidden_dim = llama_config.hidden_dim
    seq_len = llama_config.seq_len
    scaling_factor = llama_config.rope_scaling["factor"]
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
        levanter_class=LlamaRotaryEmbedding(dim=hidden_dim),
        hf_class=HFLlamaRotaryEmbedding(dim=hidden_dim, device=device),
    )

    # test LlamaLinearScalingRotaryEmbedding
    test_levanter_against_hf(
        levanter_class=LlamaLinearScalingRotaryEmbedding(dim=hidden_dim, scaling_factor=scaling_factor),
        hf_class=HFLlamaLinearScalingRotaryEmbedding(dim=hidden_dim, scaling_factor=scaling_factor, device=device),
    )

    # test LlamaDynamicNTKScalingRotaryEmbedding
    test_levanter_against_hf(
        levanter_class=LlamaDynamicNTKScalingRotaryEmbedding(dim=hidden_dim, scaling_factor=scaling_factor),
        hf_class=HFLlamaDynamicNTKScalingRotaryEmbedding(dim=hidden_dim, scaling_factor=scaling_factor, device=device),
    )
"""


def test_llama_attention():
    llama_config = _get_llama_config()
    key = random.PRNGKey(4)
    hf_llama_att = LlamaAttention.init(config=llama_config, key=key)



def _get_llama_config() -> LlamaConfig:
    vocab_size = 32000
    hidden_dim = 2048
    num_heads = 16
    num_kv_heads = 16
    rope_scaling = {
        "type": "linear",
        "factor": 2.0,
    }
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_scaling=rope_scaling,
    )
