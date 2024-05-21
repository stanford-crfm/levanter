import tempfile
import dataclasses
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import numpy as np
import pytest
from jax import random

import haliax as hax

from levanter.models.attention import AttentionMask
from levanter.models.olmo import OlmoConfig, OlmoDecoderLayer, OlmoEmbedding, OlmoLMHeadModel, OlmoMLP, OlmoTransformer
from levanter.models.olmo import _apply_rotary_pos_emb as olmo_apply_rotary_pos_emb
from levanter.models.olmo import _rotate_half as olmo_rotate_half
from levanter.models.olmo import olmo_rotary_pos_emb
from levanter.utils.jax_utils import parameter_count
from test_utils import check_load_config, check_model_works_with_seqlen, parameterize_with_configs, skip_if_no_torch


@skip_if_no_torch
def test_olmo_config():
    # load HF config and convert to levanter config
    hf_config = transformers.OlmoConfig.from_pretrained("allenai/OLMo-7B")
    olmo_config = OlmoConfig.from_hf_config(hf_config)

    # convert back to HF config
    config_overrides = {
        "_name_or_path": hf_config._name_or_path,
        "architectures": hf_config.architectures,
        "torch_dtype": hf_config.torch_dtype,
    }
    new_hf_config = olmo_config.to_hf_config(
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
def test_olmo_rotary_embedding():
    import torch
    from transformers.models.olmo.modeling_olmo import OlmoRotaryEmbedding as HFOlmoRotaryEmbedding

    olmo_config = _get_olmo_config()
    HeadSize = olmo_config.HeadSize
    Pos = olmo_config.Pos
    hidden_dim = HeadSize.size
    seq_len = Pos.size
    key = random.PRNGKey(0)
    device = "cpu"

    x = random.normal(key, (1, seq_len))
    x_torch = torch.from_numpy(np.array(x))

    levanter_output = olmo_rotary_pos_emb(HeadSize=HeadSize, Pos=Pos)
    hf_rope = HFOlmoRotaryEmbedding(dim=hidden_dim, max_position_embeddings=seq_len, device=device)
    hf_output = hf_rope(x_torch, torch.arange(seq_len).reshape(1, -1), seq_len=seq_len)

    for jax_out, torch_out in zip(levanter_output, hf_output):
        torch_out = torch_out.numpy()
        assert np.isclose(torch_out, np.array(jax_out.array), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"


@skip_if_no_torch
def test_apply_rotary_pos_emb():
    import torch
    from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
    from transformers.models.olmo.modeling_olmo import rotate_half as hf_rotate_half

    def assert_equal_out(hax_out, torch_out: torch.Tensor):
        assert np.isclose(
            torch_out.numpy(), np.array(hax_out.array), rtol=1e-2, atol=1e-2
        ).all(), f"{torch_out} != {hax_out}"

    def named_array_to_tensor(named_array):
        return torch.from_numpy(np.array(named_array.array))

    olmo_config = _get_olmo_config()

    Pos = olmo_config.Pos
    Heads = olmo_config.Heads
    HeadSize = olmo_config.HeadSize
    Batch = hax.Axis("batch", 2)

    # note here we switch Heads and Pos for the shape of the output tensors
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    k = hax.random.normal(random.PRNGKey(1), (Batch, Pos, Heads, HeadSize))

    # Check the output of _rotate_half() from levanter and hf
    levanter_out_rf_q = olmo_rotate_half(q)
    levanter_out_rf_k = olmo_rotate_half(k)

    q_tensor = named_array_to_tensor(q).transpose(1, 2)  # needed for HF
    k_tensor = named_array_to_tensor(k).transpose(1, 2)
    hf_out_rf_q = hf_rotate_half(q_tensor).transpose(1, 2)  # re-transpose to match levanter
    hf_out_rf_k = hf_rotate_half(k_tensor).transpose(1, 2)

    assert_equal_out(levanter_out_rf_q, hf_out_rf_q)
    assert_equal_out(levanter_out_rf_k, hf_out_rf_k)

    # Check the output of _apply_rotary_pos_emb() from levanter and hf
    cos = hax.random.normal(random.PRNGKey(2), (Pos, HeadSize))
    sin = hax.random.normal(random.PRNGKey(3), (Pos, HeadSize))

    levanter_out_rope_q, levanter_out_rope_k = olmo_apply_rotary_pos_emb(q, k, cos, sin)
    cos_tensor = named_array_to_tensor(cos)[None, :, :]
    sin_tensor = named_array_to_tensor(sin)[None, :, :]

    hf_out_rope_q, hf_out_rope_k = hf_apply_rotary_pos_emb(q_tensor, k_tensor, cos_tensor, sin_tensor)
    hf_out_rope_q = hf_out_rope_q.transpose(1, 2)  # re-transpose to match levanter
    hf_out_rope_k = hf_out_rope_k.transpose(1, 2)
    assert_equal_out(levanter_out_rope_q, hf_out_rope_q)
    assert_equal_out(levanter_out_rope_k, hf_out_rope_k)


# @skip_if_no_torch
# @pytest.mark.parametrize("use_flash", [True, False])
# @pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
# def test_olmo_attention(use_flash, num_kv_heads):
#     import torch
#     from transformers.models.olmo.modeling_olmo import OlmoAttention as HFOlmoAttention

#     config = _get_olmo_config(use_flash=use_flash, num_kv_heads=num_kv_heads)

#     attention = OlmoAttention.init(config=config, key=random.PRNGKey(0))

#     state = attention.to_state_dict()
#     state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
#     hf_attention = HFOlmoAttention(config.to_hf_config(32000))
#     hf_attention.load_state_dict(state, strict=True)

#     x, mask = _get_random_inputs(config)
#     x_torch = torch.from_numpy(np.array(x.array))
#     batch_size = x_torch.shape[0]
#     explicit_mask = torch.from_numpy(np.array(mask.materialize(config.Pos, config.KeyPos).array))
#     mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))

#     # the torch mask is really a bias, so we need to invert it and make it a big negative number
#     mask_torch = (mask_torch == 0).float() * -1e9

#     out = attention(x, mask)
#     position_ids = torch.arange(config.Pos.size).reshape(1, -1)
#     hf_out = hf_attention(x_torch, position_ids=position_ids, attention_mask=mask_torch)

#     assert np.isclose(
#         hf_out[0].detach().cpu().numpy(), np.array(out.array), rtol=1e-4, atol=1e-4
#     ).all(), f"{hf_out[0]} != {out}"


def test_olmo_param_counts_dont_change_with_seqlen():
    model = OlmoLMHeadModel.init(hax.Axis("v", 2048), _get_olmo_config(seq_len=128), key=random.PRNGKey(0))
    model2 = OlmoLMHeadModel.init(hax.Axis("v", 2048), _get_olmo_config(seq_len=256), key=random.PRNGKey(0))
    assert parameter_count(model) == parameter_count(model2)


@skip_if_no_torch
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo_decoder_layer(num_kv_heads):
    import torch
    from transformers.models.olmo.modeling_olmo import OlmoDecoderLayer as HFOlmoDecoderLayer

    olmo_config = _get_olmo_config(num_kv_heads=num_kv_heads)
    key = random.PRNGKey(0)
    olmo_decoder_layer = OlmoDecoderLayer.init(config=olmo_config, key=key)

    state = olmo_decoder_layer.to_state_dict()
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_decoder_layer = HFOlmoDecoderLayer(olmo_config.to_hf_config(32000), layer_idx=0)
    hf_decoder_layer.load_state_dict(state, strict=True)

    x, mask = _get_random_inputs(olmo_config)
    x_torch = torch.from_numpy(np.array(x.array))
    batch_size = x_torch.shape[0]
    explicit_mask = torch.from_numpy(np.array(mask.materialize(olmo_config.Pos, olmo_config.KeyPos).array))
    mask_torch = explicit_mask.broadcast_to((batch_size, 1, -1, -1))
    mask_torch = (mask_torch == 0).float() * -1e9

    position_ids = torch.arange(olmo_config.Pos.size).reshape(1, -1)

    out = olmo_decoder_layer(x, mask)
    hf_out = hf_decoder_layer(x_torch, position_ids=position_ids, attention_mask=mask_torch)

    assert np.isclose(
        hf_out[0].detach().cpu().numpy(), np.array(out.array), rtol=1e-4, atol=1e-4
    ).all(), f"{hf_out[0]} != {out}"


@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo_lm_head_model(num_kv_heads):
    olmo_config = _get_olmo_config(num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = olmo_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    olmo_model = OlmoLMHeadModel.init(Vocab=Vocab, config=olmo_config, key=random.PRNGKey(0))
    out = olmo_model(input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


@pytest.mark.parametrize("use_flash", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo_lm_head_model_bwd(use_flash, num_kv_heads):
    olmo_config = _get_olmo_config(use_flash=use_flash, num_kv_heads=num_kv_heads)
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 1000)
    Pos = olmo_config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()

    olmo_model = OlmoLMHeadModel.init(Vocab=Vocab, config=olmo_config, key=random.PRNGKey(0))

    def f(olmo_model, input_ids, mask):
        out = olmo_model(input_ids, mask)
        return hax.sum(out).scalar()

    _, grads = eqx.filter_value_and_grad(f)(olmo_model, input_ids, mask)


@skip_if_no_torch
@pytest.mark.parametrize("scan_layers", [True, False])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 4])
def test_olmo_roundtrip(scan_layers, num_kv_heads):
    import torch
    from transformers import AutoModelForCausalLM, OlmoForCausalLM

    converter = OlmoConfig.default_hf_checkpoint_converter

    config = OlmoConfig(
        seq_len=128,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,
        scan_layers=scan_layers,
    )
    Vocab = hax.Axis("vocab", 1000)
    hf_config = config.to_hf_config(Vocab.size)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = OlmoForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            OlmoLMHeadModel, f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
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


def _get_olmo_config(use_flash=False, num_kv_heads=4, seq_len=128) -> OlmoConfig:
    return OlmoConfig(
        seq_len=seq_len,
        hidden_dim=16,
        num_heads=4,
        num_kv_heads=num_kv_heads,
        gradient_checkpointing=False,  # disable for tests so debugging is easier
        use_flash_attention=use_flash,
        flash_attention_block_size=8 if use_flash else None,
    )


def _get_random_inputs(config: OlmoConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = AttentionMask.causal()
    return x, mask


@parameterize_with_configs("olmo*.yaml")
def test_olmo_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_pass_different_length_seq(num_kv_heads):
    config = OlmoConfig(
        seq_len=64,
        hidden_dim=64,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=num_kv_heads,
        use_flash_attention=True,
    )
    check_model_works_with_seqlen(OlmoLMHeadModel, config, 16)