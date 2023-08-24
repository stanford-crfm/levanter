import tempfile

import jax
import numpy as np
import transformers
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


def test_llama_attention():
    config = _get_llama_config()
    x, mask = _get_random_inputs(config)
    key = random.PRNGKey(0)

    attention = LlamaAttention.init(config=config, key=key)
    out = attention(x, mask)

    # assert the same shape
    assert out.array.shape == (x.axes[0].size, config.seq_len, config.hidden_dim)


@skip_if_no_torch
def test_llama_attention_vs_hf():
    import torch
    from transformers.models.llama.modeling_llama import LlamaAttention as HFLlamaAttention

    config = _get_llama_config()

    attention = LlamaAttention.init(config=config, key=random.PRNGKey(0))

    state = attention.to_state_dict()
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_attention = HFLlamaAttention(config.to_hf_config())
    hf_attention.load_state_dict(state, strict=False)

    x, mask = _get_random_inputs(config)
    x_torch = torch.from_numpy(np.array(x.array))
    mask_torch = torch.from_numpy(np.array(mask.array)).broadcast_to((2, 1, config.seq_len, config.seq_len))

    # the torch mask is really a bias, so we need to invert it and make it a big negative number
    mask_torch = (mask_torch == 0).float() * -1e9

    out = attention(x, mask)
    hf_out = hf_attention(x_torch, mask_torch)

    assert np.isclose(
        hf_out[0].detach().cpu().numpy(), np.array(out.array), rtol=1e-2, atol=1e-2
    ).all(), f"{hf_out[0]} != {out}"


def test_llama_decoder_layer():
    llama_config = _get_llama_config()
    key = random.PRNGKey(0)
    llama_decoder_layer = LlamaDecoderLayer.init(config=llama_config, key=key)
    x, mask = _get_random_inputs(llama_config)
    out = llama_decoder_layer(x, mask)
    assert out.array.shape == (x.axes[0].size, llama_config.seq_len, llama_config.hidden_dim)


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
    from transformers import AutoModelForCausalLM

    converter = LlamaConfig.default_hf_checkpoint_converter

    config = _get_llama_config()
    Vocab = hax.Axis("vocab", 1000)

    # TODO: load the first torch model with model_id from HF

    # randomly initialize a levanter model
    # TODO: use converter.load_pretrained
    model = LlamaLMHeadModel.init(
        Vocab=Vocab,
        config=config,
        key=random.PRNGKey(0),
    )

    input = hax.random.randint(random.PRNGKey(0), model.Pos, 0, model.Vocab.size)
    attn_mask = hax.nn.attention.causal_mask(model.Pos, model.config.KeyPos)

    def compute(input):
        model_output = model(input, attn_mask=attn_mask)
        return hax.nn.softmax(model_output, axis=model.Vocab)

    compute = jax.jit(compute)
    jax_out = compute(input).array

    with tempfile.TemporaryDirectory() as tmpdir:
        converter.save_pretrained(model, tmpdir)
        torch_model2 = AutoModelForCausalLM.from_pretrained(tmpdir)
        torch_model2.eval()

        torch_out2 = torch_model2(torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0))
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        assert np.isclose(torch_out2, np.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out2} != {jax_out}"


def _get_llama_config() -> LlamaConfig:
    seq_len = 128
    hidden_dim = 16
    num_heads = 4
    rope_scaling = {
        "type": "linear",
        "factor": 2.0,
    }
    return LlamaConfig(
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        rope_scaling=rope_scaling,
    )


def _get_random_inputs(config: LlamaConfig):
    Embed = config.Embed
    Pos = config.Pos
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Embed))
    mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    return x, mask
