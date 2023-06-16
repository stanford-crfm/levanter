import tempfile

import jax
import numpy as np
import pytest
from jax.random import PRNGKey
from test_utils import check_load_config, parameterize_with_configs, skip_if_no_torch
from transformers import AutoModelForCausalLM

import haliax
from levanter.models.mpt import MptConfig, MptLmHeadModel


@skip_if_no_torch
@pytest.mark.parametrize("use_bias", [True, False])
def test_mpt_nano_compare(use_bias):
    import torch

    # conjure up a fake model and compare
    vocab_size = 5257
    torch.manual_seed(0)

    # a bit hacky, using some internal-y APIs of transformers
    converter = MptConfig.default_hf_checkpoint_converter
    cls = converter.HFAutoModelClass()
    config = converter.HfConfigClass(
        d_model=32,
        max_seq_len=512,
        n_heads=8,
        n_layers=2,
        attn_config={"attn_impl": "torch", "alibi": True},
        vocab_size=vocab_size,
        no_bias=not use_bias,
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
    lev_model = lev_model.from_state_dict(loaded_checkpoint)

    hax_input = haliax.named(input, lev_config.Pos)
    causal_mask = haliax.nn.attention.causal_mask(lev_config.Pos, lev_config.KeyPos)
    with jax.disable_jit():
        lev_out = lev_model(hax_input, causal_mask, inference=True, key=None).array

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-3, rtol=1e-3)

    # now test round trip
    # convert all values to torch
    with tempfile.TemporaryDirectory() as tmpdir:
        converter._save_pretrained_local(lev_model, tmpdir)
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
#     lev_config = MptConfig.from_torch_config(config)
#
#     Vocab = haliax.Axis("vocab", config.vocab_size)
#     lev_model = MptLmHeadModel(Vocab, lev_config, key=PRNGKey(0))
#
#     lev_model = lev_model.from_state_dict(state_dict)


@parameterize_with_configs("mpt*.yaml")
def test_mpt_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)
