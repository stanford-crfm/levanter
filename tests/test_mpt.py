import tempfile

import jax
import numpy as np
import pytest
from jax.random import PRNGKey
from transformers import AutoModelForCausalLM

import haliax

from levanter.models.mpt import MptConfig, MptLmHeadModel
from levanter.utils.tree_utils import inference_mode
from test_utils import check_model_works_with_seqlen, skip_if_no_torch


@pytest.mark.skip(reason="MPT is broken in the latest version of transformers")
@skip_if_no_torch
@pytest.mark.parametrize("attn_impl", ["torch", "flash"])
def test_mpt_nano_compare(attn_impl):
    import torch

    # conjure up a fake model and compare
    vocab_size = 5257
    torch.manual_seed(0)

    # a bit hacky, using some internal-y APIs of transformers
    converter = MptConfig().hf_checkpoint_converter()
    cls = converter.HFAutoModelClass()
    config = converter.HfConfigClass(
        d_model=32,
        max_seq_len=512,
        n_heads=8,
        n_layers=2,
        attn_config={"attn_impl": attn_impl, "alibi": True, "flash_attention_block_size": 32},
        vocab_size=vocab_size,
    )

    model = cls(config)

    # conjure a fake input
    input = jax.random.randint(PRNGKey(0), (512,), 0, vocab_size)
    input_torch = torch.from_numpy(np.array(input)).to(torch.int64).unsqueeze(0)

    # run the model
    model.eval()
    with torch.no_grad():
        torch_out = model(input_torch)
        torch_out = torch_out.logits[0].detach().cpu().numpy()

    # now compare levanter
    with tempfile.TemporaryDirectory() as tmpdir:
        lev_config = converter.config_from_hf_config(config)
        model.save_pretrained(tmpdir)
        loaded_checkpoint = converter.load_state_dict(tmpdir)

    roundtrip_hf_config = converter.hf_config_from_config(lev_config)

    for k, v in roundtrip_hf_config.__dict__.items():
        assert getattr(roundtrip_hf_config, k) == v, f"{k} {getattr(roundtrip_hf_config, k)} != {v}"

    Vocab = haliax.Axis("vocab", vocab_size)
    lev_model = MptLmHeadModel.init(Vocab, lev_config, key=PRNGKey(0))
    lev_model = inference_mode(lev_model, True)
    lev_model = lev_model.from_state_dict(loaded_checkpoint)

    hax_input = haliax.named(input, lev_config.Pos)
    causal_mask = haliax.AttentionMask.causal()
    lev_out = lev_model(hax_input, causal_mask).array

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-3, rtol=1e-3)

    # now test round trip
    # convert all values to torch
    with tempfile.TemporaryDirectory() as tmpdir:
        # FA hack: we need to override the config to use torch attention, as their impl doesn't work with flash/alibi
        converter = converter.with_config_overrides(config_overrides={"attn_config": {"attn_impl": "torch"}})
        converter._save_pretrained_local(
            lev_model, tmpdir, save_tokenizer=True, save_reference_code=True, max_shard_size=1e8
        )
        model = AutoModelForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)

    model.eval()
    with torch.no_grad():
        torch_out = model(input_torch)
        torch_out = torch_out.logits[0].detach().cpu().numpy()

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-3, rtol=1e-3)


# @skip_if_no_torch
# def test_load_full_mpt():
#     model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b", trust_remote_code=True)
#
#     state_dict = model.state_dict()
#     # move to cpu
#     state_dict = {k: v.cpu() for k, v in state_dict.items()}
#     config = model.config
#
#     del model
#
#     lev_config = MptConfig.from_hf_config(config)
#
#     Vocab = haliax.Axis("vocab", config.vocab_size)
#     lev_model = MptLmHeadModel.from_hf_pretrained("mosaicml/mpt-7b")


def test_pass_different_length_seq():
    config = MptConfig(
        max_seq_len=32,
        d_model=16,
        n_layers=4,
        n_heads=2,
    )
    check_model_works_with_seqlen(MptLmHeadModel, config, 16)
