import os
import tempfile

import jax
import numpy as np
import pytest
from jax import random

import haliax as hax
from haliax import Axis

from levanter.models.gpt_oss import GptOssConfig, GptOssLMHeadModel
from test_utils import skip_if_no_torch


@skip_if_no_torch
def test_gpt_oss_config():
    import hf_gpt_oss

    hf_config = hf_gpt_oss.GptOssConfig(
        num_hidden_layers=2,
        num_local_experts=4,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=32,
        rope_theta=10000.0,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.1,
        output_router_logits=False,
        layer_types=["full_attention", "full_attention"],
    )

    cfg = GptOssConfig.from_hf_config(hf_config)
    new_hf_config = cfg.to_hf_config(hf_config.vocab_size)

    for k in [
        "num_hidden_layers",
        "num_local_experts",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "sliding_window",
        "num_experts_per_tok",
    ]:
        assert getattr(hf_config, k) == getattr(new_hf_config, k)


@skip_if_no_torch
def test_gpt_oss_roundtrip():
    import torch
    import hf_gpt_oss

    config = GptOssConfig(
        seq_len=64,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
    )
    Vocab = Axis("vocab", 100)
    hf_config = config.to_hf_config(Vocab.size)

    input_ids = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)
    input_torch = torch.from_numpy(np.array(input_ids.array)).to(torch.int32).unsqueeze(0)

    torch_model = hf_gpt_oss.GptOssForCausalLM(hf_config)
    torch_model.eval()
    torch_logits = torch_model(input_torch).logits[0].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        converter = config.hf_checkpoint_converter(
            ref_checkpoint=os.path.dirname(hf_gpt_oss.__file__),
            tokenizer="hf-internal-testing/llama-tokenizer",
        )
        model = converter.load_pretrained(
            GptOssLMHeadModel, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        def compute(input_ids):
            return model(input_ids, attn_mask=attn_mask)

        jax_logits = compute(input_ids).array
        assert jax_logits.shape == torch_logits.shape
        np.testing.assert_allclose(torch_logits, np.array(jax_logits), rtol=1e-4, atol=1e-4)

        converter.save_pretrained(
            model, f"{tmpdir}/lev_model", save_reference_code=True, save_tokenizer=False
        )
        torch_model2 = hf_gpt_oss.GptOssForCausalLM.from_pretrained(
            f"{tmpdir}/lev_model", trust_remote_code=True
        )
        torch_model2.eval()
        torch_logits2 = torch_model2(input_torch).logits[0].detach().cpu().numpy()
        np.testing.assert_allclose(torch_logits2, np.array(jax_logits), rtol=1e-4, atol=1e-4)


@skip_if_no_torch
def test_state_dict_consistency():
    import hf_gpt_oss

    config = GptOssConfig(
        seq_len=64,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        gradient_checkpointing=False,
        tie_word_embeddings=False,
    )
    Vocab = Axis("vocab", 100)
    model = GptOssLMHeadModel.init(Vocab=Vocab, config=config, key=random.PRNGKey(0))
    hf_model = hf_gpt_oss.GptOssForCausalLM(config.to_hf_config(Vocab.size))
    lev_state_dict = hax.state_dict.to_torch_compatible_state_dict(model)
    assert set(hf_model.state_dict().keys()) == set(lev_state_dict.keys())

