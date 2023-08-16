import numpy as np
import torch
from jax import random
from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention as HFLlamaAttention
from transformers.models.llama.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding as HFLlamaDynamicNTKScalingRotaryEmbedding,
)
from transformers.models.llama.modeling_llama import (
    LlamaLinearScalingRotaryEmbedding as HFLlamaLinearScalingRotaryEmbedding,
)
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import rotate_half as hf_rotate_half

import haliax as hax

from levanter.models.llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaLMHeadModel,
    LlamaRotaryEmbedding,
)
from levanter.models.llama import _apply_rotary_pos_emb as levanter_apply_rotary_pos_emb
from levanter.models.llama import _rotate_half as levanter_rotate_half


def test_llama_rotary_embedding():
    llama_config = _get_llama_config()
    Embed = llama_config.Embed
    Pos = llama_config.Pos
    hidden_dim = Embed.size
    seq_len = Pos.size
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
        levanter_class=LlamaRotaryEmbedding(Embed=Embed, Pos=Pos),
        hf_class=HFLlamaRotaryEmbedding(dim=hidden_dim, max_position_embeddings=seq_len, device=device),
    )

    # test LlamaLinearScalingRotaryEmbedding
    test_levanter_against_hf(
        levanter_class=LlamaLinearScalingRotaryEmbedding(Embed=Embed, Pos=Pos, scaling_factor=scaling_factor),
        hf_class=HFLlamaLinearScalingRotaryEmbedding(
            dim=hidden_dim, max_position_embeddings=seq_len, scaling_factor=scaling_factor, device=device
        ),
    )

    # test LlamaDynamicNTKScalingRotaryEmbedding
    test_levanter_against_hf(
        levanter_class=LlamaDynamicNTKScalingRotaryEmbedding(Embed=Embed, Pos=Pos, scaling_factor=scaling_factor),
        hf_class=HFLlamaDynamicNTKScalingRotaryEmbedding(
            dim=hidden_dim, max_position_embeddings=seq_len, scaling_factor=scaling_factor, device=device
        ),
    )


def test_apply_rotary_pos_emb():
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


def test_llama_attention():
    config = _get_llama_config()
    x, mask, position_ids = _get_random_inputs(config)
    # generate a random key that can be splitted into 4
    key = random.PRNGKey(4)

    levanter_attention = LlamaAttention.init(config=config, key=key)
    levanter_out = levanter_attention(x, mask, position_ids)

    hf_config = _levanter_config_to_hf_config(config)
    hf_attention = HFLlamaAttention(config=hf_config)  # (seq_len, kv_seq_len)
    # convert attention_mask's shape from (seq_len, kv_seq_len) to (batch, 1, seq_len, kv_seq_len)
    attention_mask = _hax_to_tensor(mask)
    attention_mask = attention_mask.reshape(1, 1, config.Pos.size, config.KeyPos.size).repeat(x.axes[0].size, 1, 1, 1)

    hf_out, _, _ = hf_attention(
        hidden_states=_hax_to_tensor(x),
        attention_mask=attention_mask,
        position_ids=torch.from_numpy(np.array(position_ids)),
    )

    # assert the same shape
    assert levanter_out.array.shape == hf_out.shape, f"{levanter_out.shape} != {hf_out.shape}"


def test_llama_decoder_layer():
    llama_config = _get_llama_config()
    key = random.PRNGKey(0)
    llama_decoder_layer = LlamaDecoderLayer.init(config=llama_config, key=key)
    x, mask, position_ids = _get_random_inputs(llama_config)
    levanter_out = llama_decoder_layer(x, mask, position_ids)
    assert levanter_out.array.shape == (x.axes[0].size, llama_config.seq_len, llama_config.hidden_dim)


def test_llama_lm_head_model():
    llama_config = _get_llama_config()
    Vocab = hax.Axis("vocab", llama_config.vocab_size)
    # generate a key that can be splitted into 2
    llama_model = LlamaLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))
    # generate a random input
    x, mask, position_ids = _get_random_inputs(llama_config)

    levanter_out = llama_model(x, mask, position_ids)
    assert levanter_out.array.shape == (Batch.size, Pos.size, llama_config.Vocab.size)


def _assert_equal_out(hax_out, torch_out: torch.Tensor):
    assert np.isclose(
        torch_out.numpy(), np.array(hax_out.array), rtol=1e-2, atol=1e-2
    ).all(), f"{torch_out} != {hax_out}"


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
    position_ids = random.randint(random.PRNGKey(2), (Batch.size, Pos.size), 0, Pos.size)
    return x, mask, position_ids


def _levanter_config_to_hf_config(levanter_config: LlamaConfig) -> HFLlamaConfig:
    return HFLlamaConfig(
        vocab_size=levanter_config.vocab_size,
        max_position_embeddings=levanter_config.seq_len,
        hidden_size=levanter_config.hidden_dim,
        num_attention_heads=levanter_config.num_heads,
        num_key_value_heads=levanter_config.num_kv_heads,
        rope_scaling=levanter_config.rope_scaling,
    )


def _hax_to_tensor(x: hax.NamedArray) -> torch.Tensor:
    return torch.from_numpy(np.array(x.array))
