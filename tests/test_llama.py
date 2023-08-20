import numpy as np
from jax import random

import haliax as hax

from levanter.models.llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaLMHeadModel,
    LlamaRotaryEmbedding,
)
from levanter.models.llama import _apply_rotary_pos_emb as levanter_apply_rotary_pos_emb
from levanter.models.llama import _rotate_half as levanter_rotate_half
from test_utils import skip_if_no_torch


@skip_if_no_torch
def test_llama_rotary_embedding():
    import torch
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

    llama_config = _get_llama_config()
    HeadSize = llama_config.HeadSize
    Pos = llama_config.Pos
    hidden_dim = HeadSize.size
    seq_len = Pos.size
    key = random.PRNGKey(0)
    device = "cpu"

    x = random.normal(key, (1, seq_len))
    x_torch = torch.from_numpy(np.array(x))

    levanter_rope = LlamaRotaryEmbedding(HeadSize=HeadSize, Pos=Pos)
    levanter_output = levanter_rope(seq_len=seq_len)
    hf_rope = HFLlamaRotaryEmbedding(dim=hidden_dim, max_position_embeddings=seq_len, device=device)
    hf_output = hf_rope(x_torch, seq_len=seq_len)

    for jax_out, torch_out in zip(levanter_output, hf_output):
        torch_out = torch_out.numpy()
        assert np.isclose(torch_out, np.array(jax_out.array), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"


@skip_if_no_torch
def test_apply_rotary_pos_emb():
    import torch
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
    from transformers.models.llama.modeling_llama import rotate_half as hf_rotate_half

    def assert_equal_out(hax_out, torch_out: torch.Tensor):
        assert np.isclose(
            torch_out.numpy(), np.array(hax_out.array), rtol=1e-2, atol=1e-2
        ).all(), f"{torch_out} != {hax_out}"

    def named_array_to_tensor(named_array):
        return torch.from_numpy(np.array(named_array.array))

    llama_config = _get_llama_config()

    Pos = llama_config.Pos
    Heads = llama_config.Heads
    HeadSize = llama_config.HeadSize
    Batch = hax.Axis("batch", 2)

    # note here we switch Heads and Pos for the shape of the output tensors
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    k = hax.random.normal(random.PRNGKey(1), (Batch, Pos, Heads, HeadSize))

    # Check the output of _rotate_half() from levanter and hf
    levanter_out_rf_q = levanter_rotate_half(q)
    levanter_out_rf_k = levanter_rotate_half(k)

    q_tensor = named_array_to_tensor(q).transpose(1, 2)  # needed for HF
    k_tensor = named_array_to_tensor(k).transpose(1, 2)
    hf_out_rf_q = hf_rotate_half(q_tensor).transpose(1, 2)  # re-transpose to match levanter
    hf_out_rf_k = hf_rotate_half(k_tensor).transpose(1, 2)

    assert_equal_out(levanter_out_rf_q, hf_out_rf_q)
    assert_equal_out(levanter_out_rf_k, hf_out_rf_k)

    # Check the output of _apply_rotary_pos_emb() from levanter and hf
    cos = hax.random.normal(random.PRNGKey(2), (Pos, HeadSize))
    sin = hax.random.normal(random.PRNGKey(3), (Pos, HeadSize))
    position_ids = hax.arange(Pos).broadcast_axis(Batch)

    levanter_out_rope_q, levanter_out_rope_k = levanter_apply_rotary_pos_emb(Pos, q, k, cos, sin, position_ids)
    cos_tensor = named_array_to_tensor(cos)
    sin_tensor = named_array_to_tensor(sin)
    position_ids_tensor = named_array_to_tensor(position_ids)

    hf_out_rope_q, hf_out_rope_k = hf_apply_rotary_pos_emb(
        q_tensor, k_tensor, cos_tensor, sin_tensor, position_ids_tensor
    )
    hf_out_rope_q = hf_out_rope_q.transpose(1, 2)  # re-transpose to match levanter
    hf_out_rope_k = hf_out_rope_k.transpose(1, 2)
    assert_equal_out(levanter_out_rope_q, hf_out_rope_q)
    assert_equal_out(levanter_out_rope_k, hf_out_rope_k)


def test_llama_attention():
    config = _get_llama_config()
    x, mask, position_ids = _get_random_inputs(config)
    # generate a random key that can be splitted into 4
    key = random.PRNGKey(4)

    attention = LlamaAttention.init(config=config, key=key)
    out = attention(x, mask, position_ids)

    # assert the same shape
    assert out.array.shape == (x.axes[0].size, config.seq_len, config.hidden_dim)


def test_llama_decoder_layer():
    llama_config = _get_llama_config()
    key = random.PRNGKey(0)
    llama_decoder_layer = LlamaDecoderLayer.init(config=llama_config, key=key)
    x, mask, position_ids = _get_random_inputs(llama_config)
    out = llama_decoder_layer(x, mask, position_ids)
    assert out.array.shape == (x.axes[0].size, llama_config.seq_len, llama_config.hidden_dim)


def test_llama_lm_head_model():
    llama_config = _get_llama_config()
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", llama_config.vocab_size)
    Pos = llama_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, llama_config.vocab_size)
    mask = hax.nn.attention.causal_mask(Pos, llama_config.KeyPos)

    llama_model = LlamaLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))
    out = llama_model(input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


def _get_llama_config() -> LlamaConfig:
    vocab_size = 1000
    seq_len = 128
    hidden_dim = 16
    num_heads = 4
    num_kv_heads = 4
    rope_scaling = {
        "type": "linear",
        "factor": 2.0,
    }
    return LlamaConfig(
        seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        rope_scaling=rope_scaling,
    )


def _get_random_inputs(config: LlamaConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    position_ids = hax.arange(Pos).broadcast_axis(Batch)
    return x, mask, position_ids
