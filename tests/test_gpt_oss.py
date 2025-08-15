import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
import haliax as hax
import pytest
from jax import random

from levanter.layers.attention import AttentionMask
from levanter.models.gpt_oss import (
    GptOssConfig,
    GptOssLMHeadModel,
    GptOssSparseMoeBlock,
)
from test_utils import skip_if_no_torch


def _get_config(**overrides):
    config = GptOssConfig(
        seq_len=8,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return dataclasses.replace(config, **overrides) if overrides else config


@skip_if_no_torch
def test_gpt_oss_config_roundtrip():
    from hf_gpt_oss import GptOssConfig as HfGptOssConfig

    hf_config = HfGptOssConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        output_router_logits=False,
        sliding_window=None,
    )
    config = GptOssConfig.from_hf_config(hf_config)
    new_hf = config.to_hf_config(vocab_size=hf_config.vocab_size)
    for field in [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_local_experts",
        "num_experts_per_tok",
        "router_aux_loss_coef",
        "output_router_logits",
        "sliding_window",
    ]:
        assert getattr(new_hf, field) == getattr(hf_config, field)


def test_gpt_oss_moe_block():
    if not hasattr(hax, "shard_map"):
        pytest.skip("hax.shard_map not available")
    config = _get_config(num_layers=1)
    block = GptOssSparseMoeBlock.init(config, key=random.PRNGKey(0))
    Batch = hax.Axis("batch", 2)
    x = hax.random.normal(random.PRNGKey(1), (Batch, config.Pos, config.Embed))
    out, extras = block(x)
    assert out.axes == x.axes
    assert "expert_loads" in extras
    if config.router_aux_loss_coef is not None:
        assert "load_balancing_loss" in extras


@skip_if_no_torch
def test_gpt_oss_hf_serialization():
    import torch
    from hf_gpt_oss import GptOssForCausalLM

    config = _get_config(num_layers=1)
    Vocab = hax.Axis("vocab", 32)
    model = GptOssLMHeadModel.init(Vocab, config, key=random.PRNGKey(0))
    hf_config = config.to_hf_config(vocab_size=Vocab.size)
    state = hax.state_dict.to_torch_compatible_state_dict(model)
    state = {k: torch.from_numpy(np.array(v)) for k, v in state.items()}
    hf_model = GptOssForCausalLM(hf_config)
    hf_model.load_state_dict(state, strict=True)

    Batch = hax.Axis("batch", 1)
    input_ids = hax.arange(config.Pos, dtype=jnp.int32).broadcast_axis(Batch)
    mask = AttentionMask.causal()
    logits = model(input_ids, attn_mask=mask, key=random.PRNGKey(1))
    hf_logits = hf_model(torch.tensor(input_ids.array)).logits
    np.testing.assert_allclose(logits.array, hf_logits.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)
