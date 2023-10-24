import tempfile

import jax
import numpy as np
import pytest
import transformers
from jax import random

import haliax as hax

from levanter.models.llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaLMHeadModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from levanter.models.llama import _apply_rotary_pos_emb as levanter_apply_rotary_pos_emb
from levanter.models.llama import _rotate_half as levanter_rotate_half
from test_utils import check_load_config, check_model_works_with_seqlen, parameterize_with_configs, skip_if_no_torch


@skip_if_no_torch
def test_llama_config():
    # load HF config and convert to levanter config
    hf_config = transformers.LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    llama_config = LlamaConfig.from_hf_config(hf_config)

    # convert back to HF config
    config_overrides = {
        "_name_or_path": hf_config._name_or_path,
        "architectures": hf_config.architectures,
        "torch_dtype": hf_config.torch_dtype,
    }
    new_hf_config = llama_config.to_hf_config(
        vocab_size=hf_config.vocab_size,
        config_overrides=config_overrides,
    )

    # assert the content in new_hf_config is the same as hf_config
    for k in new_hf_config.__dict__.keys():
        if k in ["_commit_hash", "transformers_version"]:
            continue
        assert getattr(new_hf_config, k) == getattr(
            hf_config, k
        ), f"{k} {getattr(new_hf_config, k)} != {getattr(hf_config, k)}"


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

    levanter_out_rope_q, levanter_out_rope_k = levanter_apply_rotary_pos_emb(q, k, cos, sin)
    cos_tensor = named_array_to_tensor(cos)
    sin_tensor = named_array_to_tensor(sin)
    position_ids = hax.arange(Pos).broadcast_axis(Batch)
    position_ids_tensor = named_array_to_tensor(position_ids)

    hf_out_rope_q, hf_out_rope_k = hf_apply_rotary_pos_emb(
        q_tensor, k_tensor, cos_tensor, sin_tensor, position_ids_tensor
    )
    hf_out_rope_q = hf_out_rope_q.transpose(1, 2)  # re-transpose to match levanter
    hf_out_rope_k = hf_out_rope_k.transpose(1, 2)
    assert_equal_out(levanter_out_rope_q, hf_out_rope_q)
    assert_equal_out(levanter_out_rope_k, hf_out_rope_k)


@skip_if_no_torch
@pytest.mark.parametrize("use_flash", [True, False])
def test_llama_attention(use_flash):
    import torch
    from transformers.models.llama.modeling_llama import LlamaAttention as HFLlamaAttention

    config = _get_llama_config(use_flash=use_flash)

    attention = LlamaAttention.init(config=config, key=random.PRNGKey(0))

    state = attention.to_state_dict()
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_attention = HFLlamaAttention(config.to_hf_config(32000))
    hf_attention.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    mask_torch = torch.from_numpy(np.array(mask.array)).broadcast_to((batch_size, 1, -1, -1))

    # the torch mask is really a bias, so we need to invert it and make it a big negative number
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    hf_out = hf_attention(x_torch, mask_torch)

    assert np.isclose(
        hf_out[0].detach().cpu().numpy(), np.array(out.array), rtol=1e-4, atol=1e-4
    ).all(), f"{hf_out[0]} != {out}"


@skip_if_no_torch
def test_llama_rms_norm():
    import torch
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFLlamaRMSNorm

    config = _get_llama_config()
    ln = LlamaRMSNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
    hf_ln = HFLlamaRMSNorm(config.Embed.size, eps=config.layer_norm_epsilon)

    x, _ = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))

    out = ln(x)
    hf_out = hf_ln(x_torch)

    assert np.isclose(
        hf_out.detach().cpu().numpy(), np.array(out.array), rtol=1e-6, atol=1e-6
    ).all(), f"{hf_out} != {out}"


@skip_if_no_torch
def test_llama_decoder_layer():
    import torch
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFLlamaDecoderLayer

    llama_config = _get_llama_config()
    key = random.PRNGKey(0)
    llama_decoder_layer = LlamaDecoderLayer.init(config=llama_config, key=key)

    state = llama_decoder_layer.to_state_dict()
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_decoder_layer = HFLlamaDecoderLayer(llama_config.to_hf_config(32000))
    hf_decoder_layer.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(llama_config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    mask_torch = torch.from_numpy(np.array(mask.array)).broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e9

    out = llama_decoder_layer(x, mask)
    hf_out = hf_decoder_layer(x_torch, mask_torch)

    assert np.isclose(
        hf_out[0].detach().cpu().numpy(), np.array(out.array), rtol=1e-4, atol=1e-4
    ).all(), f"{hf_out[0]} != {out}"


def test_llama_lm_head_model():
    llama_config = _get_llama_config()
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = llama_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = hax.nn.attention.causal_mask(Pos, llama_config.KeyPos)

    llama_model = LlamaLMHeadModel.init(Vocab=Vocab, config=llama_config, key=random.PRNGKey(0))
    out = llama_model(input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


@skip_if_no_torch
def test_llama_roundtrip():
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    converter = LlamaConfig.default_hf_checkpoint_converter

    config = LlamaConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        gradient_checkpointing=False,
    )
    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = LlamaForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            LlamaLMHeadModel, f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        def compute(input):
            model_output = model(input, attn_mask=attn_mask)
            return hax.nn.softmax(model_output, axis=model.Vocab)

        compute = jax.jit(compute)
        jax_out = compute(input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        assert np.isclose(torch_out2, np.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out2} != {jax_out}"


def _get_llama_config(use_flash=False) -> LlamaConfig:
    rope_scaling = {
        "type": "linear",
        "factor": 2.0,
    }
    return LlamaConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        rope_scaling=rope_scaling,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
    )


def _get_random_inputs(config: LlamaConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    return x, mask


@parameterize_with_configs("llama*.yaml")
def test_llama_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)


def test_pass_different_length_seq():
    config = LlamaConfig(
        seq_len=32,
        hidden_dim=16,
        intermediate_dim=32,
        num_heads=2,
    )
    check_model_works_with_seqlen(LlamaLMHeadModel, config, 16)
