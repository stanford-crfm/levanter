import numpy as np

import haliax as hax
import jax.numpy as jnp
import torch
from jax import random

# The latter 2 classes are only available in HuggingFace's transformers 4.30.0 or later
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding as HFLlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding as HFLlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding as HFLlamaDynamicNTKScalingRotaryEmbedding,
    LlamaAttention as HFLlamaAttention,
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    rotate_half as hf_rotate_half,
)
from levanter.models.llama import (
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaAttention,
    _apply_rotary_pos_emb as levanter_apply_rotary_pos_emb,
    _rotate_half as levanter_rotate_half,
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



def test_llama_attention():
    llama_config = _get_llama_config()
    key = random.PRNGKey(4)
    levanter_llama_att = LlamaAttention.init(config=llama_config, key=key)
    seq_len = llama_config.seq_len

    input_ids = hax.arange(llama_config.Pos, dtype=jnp.int32)
    causal_mask = hax.nn.attention.causal_mask(llama_config.Pos, llama_config.KeyPos)
    position_ids = random.randint(random.PRNGKey(0), (1, seq_len), 0, llama_config.Pos.size)
    levanter_output = levanter_llama_att(input_ids, mask=causal_mask, position_ids=position_ids)
"""


def test_apply_rotary_pos_emb():
    llama_config = _get_llama_config()

    Pos = llama_config.Pos
    Heads = llama_config.Heads
    KVHeads = llama_config.KVHeads
    HeadSize = llama_config.HeadSize
    Batch = hax.Axis("batch", 2)

    # note here we switch Heads and Pos for the shape of the output tensors
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    k = hax.random.normal(random.PRNGKey(1), (Batch, Pos, Heads, HeadSize))

    # Check the output of _rotate_half() from levanter and hf
    levanter_out_rf_q = levanter_rotate_half(q)
    levanter_out_rf_k = levanter_rotate_half(k)

    q_tensor = torch.from_numpy(np.array(q.array)).transpose(1, 2)  # needed for HF
    k_tensor = torch.from_numpy(np.array(k.array)).transpose(1, 2)
    hf_out_rf_q = hf_rotate_half(q_tensor).transpose(1, 2)  # re-transpose to match levanter
    hf_out_rf_k = hf_rotate_half(k_tensor).transpose(1, 2)

    _assert_equal_out(levanter_out_rf_q, hf_out_rf_q)
    _assert_equal_out(levanter_out_rf_k, hf_out_rf_k)

    # Check the output of _apply_rotary_pos_emb() from levanter and hf
    cos = random.normal(random.PRNGKey(2), (1, 1, Pos.size, HeadSize.size))
    sin = random.normal(random.PRNGKey(3), (1, 1, Pos.size, HeadSize.size))
    position_ids = random.randint(random.PRNGKey(4), (Batch.size, Pos.size), 0, Pos.size)

    levanter_out_rope_q, levanter_out_rope_k = levanter_apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    cos_tensor = torch.from_numpy(np.array(cos))
    sin_tensor = torch.from_numpy(np.array(sin))
    position_ids_tensor = torch.from_numpy(np.array(position_ids))

    hf_out_rope_q, hf_out_rope_k = hf_apply_rotary_pos_emb(
        q_tensor, k_tensor, cos_tensor, sin_tensor, position_ids_tensor
    )
    hf_out_rope_q = hf_out_rope_q.transpose(1, 2)  # re-transpose to match levanter
    hf_out_rope_k = hf_out_rope_k.transpose(1, 2)
    _assert_equal_out(levanter_out_rope_q, hf_out_rope_q)
    _assert_equal_out(levanter_out_rope_k, hf_out_rope_k)


def _assert_equal_out(hax_out, torch_out: torch.Tensor):
    assert np.isclose(
        torch_out.numpy(), np.array(hax_out.array), rtol=1e-2, atol=1e-2
    ).all(), f"{torch_out} != {hax_out}"


def _get_llama_config() -> LlamaConfig:
    vocab_size = 32000
    hidden_dim = 48
    num_heads = 8
    num_kv_heads = 8
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
