import tempfile

import jax
import numpy as np
from jax.random import PRNGKey
from transformers import AutoModelForCausalLM

import haliax
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

from levanter.models.attention import AttentionMask
from levanter.models.backpack import BackpackConfig, BackpackLMHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode
from test_utils import check_load_config, check_model_works_with_seqlen, parameterize_with_configs, skip_if_no_torch


VOCAB_SIZE = 50264


def test_backpack_predict():
    trainer_config = TrainerConfig()

    Vocab = round_axis_for_partitioning(Axis("vocab", VOCAB_SIZE), trainer_config.compute_axis_mapping)
    model_config = BackpackConfig(use_flash_attention=False)
    model_key = PRNGKey(0)
    model = BackpackLMHeadModel.init(Vocab, model_config, key=model_key)
    mp = trainer_config.mp
    model = mp.cast_to_param(model)

    input = hax.random.randint(PRNGKey(0), model.Pos, 0, model.Vocab.size)
    attn_mask = AttentionMask.causal()

    def compute(input):
        return hax.nn.softmax(
            model(input, key=None, attn_mask=attn_mask),
            axis=model.Vocab,
        )

    out = compute(input).array
    assert out.shape == (
        model.Pos.size,
        model.Vocab.size,
    ), f"{out.shape} != {(model.Pos, model.Vocab.size)}"


@skip_if_no_torch
def test_backpack_nano_compare():
    import torch

    # conjure up a fake model and compare
    vocab_size = 5257
    torch.manual_seed(0)

    converter = BackpackConfig().hf_checkpoint_converter()

    # a bit hacky, using some internal-y APIs of transformers
    cls = converter.HFAutoModelClass()
    config = converter.HfConfigClass(
        n_embd=32,
        n_positions=512,
        n_head=8,
        n_layer=2,
        vocab_size=vocab_size,
        resid_pdrop=0.0,
    )

    model = cls(config)
    model.tie_weights()

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
        model.save_pretrained(tmpdir, safe_serialization=False)  # unsafe b/c weight tying
        loaded_checkpoint = converter.load_state_dict(tmpdir)

    roundtrip_hf_config = converter.hf_config_from_config(lev_config)

    for k, v in roundtrip_hf_config.__dict__.items():
        assert getattr(roundtrip_hf_config, k) == v, f"{k} {getattr(roundtrip_hf_config, k)} != {v}"

    Vocab = haliax.Axis("vocab", vocab_size)
    lev_model = BackpackLMHeadModel.init(Vocab, lev_config, key=PRNGKey(0))
    lev_model = haliax.state_dict.from_torch_compatible_state_dict(lev_model, loaded_checkpoint)
    lev_model = inference_mode(lev_model, True)

    hax_input = haliax.named(input, lev_config.Pos)
    attn_mask = hax.nn.attention.causal_mask(lev_config.Pos, lev_config.KeyPos)
    with jax.disable_jit():
        lev_out = lev_model(hax_input, attn_mask=attn_mask, key=None).array

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-2, rtol=1e-2)

    # now test round trip
    with tempfile.TemporaryDirectory() as tmpdir:
        converter.save_pretrained(lev_model, tmpdir)
        model = AutoModelForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)

    model.eval()
    with torch.no_grad():
        torch_out = model(input_torch)
        torch_out = torch_out.logits[0].detach().cpu().numpy()

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-3, rtol=1e-3)


@parameterize_with_configs("backpack*.yaml")
def test_backpack_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    config_class = TrainLmConfig

    check_load_config(config_class, config_file)


def test_pass_different_length_seq():
    config = BackpackConfig(
        seq_len=64,
        hidden_dim=16,
        num_layers=4,
        num_heads=2,
        gradient_checkpointing=False,
        use_flash_attention=True,
    )
    check_model_works_with_seqlen(BackpackLMHeadModel, config, 16)
