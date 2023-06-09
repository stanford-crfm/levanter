import glob
import os
import tempfile

import jax
import numpy as np
from jax.random import PRNGKey
from test_utils import skip_if_no_torch

import haliax
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.config import TrainerConfig
from levanter.models.backpack import BackpackConfig, BackpackLMHeadModel


VOCAB_SIZE = 50264


def test_backpack_predict():
    trainer_config = TrainerConfig()

    Vocab = round_axis_for_partitioning(Axis("vocab", VOCAB_SIZE), trainer_config.compute_axis_mapping)
    model_config = BackpackConfig()
    model_key = PRNGKey(0)
    model = BackpackLMHeadModel.init(Vocab, model_config, key=model_key)
    mp = trainer_config.mp
    model = mp.cast_to_param(model)

    input = hax.random.randint(PRNGKey(0), model.Pos, 0, model.Vocab.size)
    attn_mask = hax.nn.attention.causal_mask(model.Pos, model.config.KeyPos)

    def compute(input):
        return hax.nn.softmax(
            model(input, inference=True, key=None, attn_mask=attn_mask),
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

    converter = HFCheckpointConverter(BackpackConfig, "stanford-crfm/levanter-backpack-1b", trust_remote_code=True)

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
        model.save_pretrained(tmpdir)
        loaded_checkpoint = converter.load_state_dict(tmpdir)

    roundtrip_hf_config = converter.hf_config_from_config(lev_config)

    for k, v in roundtrip_hf_config.__dict__.items():
        assert getattr(roundtrip_hf_config, k) == v, f"{k} {getattr(roundtrip_hf_config, k)} != {v}"

    Vocab = haliax.Axis("vocab", vocab_size)
    lev_model = BackpackLMHeadModel.init(Vocab, lev_config, key=PRNGKey(0))
    lev_model = lev_model.from_state_dict(loaded_checkpoint)

    hax_input = haliax.named(input, lev_config.Pos)
    attn_mask = hax.nn.attention.causal_mask(lev_config.Pos, lev_config.KeyPos)
    with jax.disable_jit():
        lev_out = lev_model(hax_input, attn_mask=attn_mask, inference=True, key=None).array

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-2, rtol=1e-2)

    # now test round trip
    lev_model = lev_model.to_state_dict()

    # convert all values to torch
    for k, v in lev_model.items():
        lev_model[k] = torch.from_numpy(np.array(v))

    model = cls(config)
    model.load_state_dict(lev_model, strict=False)

    # # TODO: switch to HF serialization in this test
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     _save_hf_checkpoint_local(lev_model, tmpdir, model_type="backpack-gpt2", auto_map_config=HFAutoMapConfig(
    #         AutoConfig="backpack_config.BackpackGPT2Config",
    #         AutoModelForCausalLM="backpack_model.BackpackGPT2LMHeadModel"
    #     ))
    #     model = AutoModelForCausalLM.from_pretrained(tmpdir, trust_remote_code=True)

    model.eval()
    with torch.no_grad():
        torch_out = model(input_torch)
        torch_out = torch_out.logits[0].detach().cpu().numpy()

    np.testing.assert_allclose(torch_out, np.array(lev_out), atol=1e-3, rtol=1e-3)


def test_backpack_configs():
    # load the TrainbackpackConfig from ../examples/backpack_example.py
    test_path = os.path.dirname(os.path.abspath(__file__))
    backpack_configs = os.path.join(test_path, "..", "config")

    # load module. might not be in pythonpath so we have to do this
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "backpack_example", os.path.join(test_path, "..", "examples", "backpack_example.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TrainBackpackConfig = module.TrainBackpackConfig

    for config_file in glob.glob(os.path.join(backpack_configs, "backpack*.yaml")):
        try:
            import pyrallis

            pyrallis.parse(TrainBackpackConfig, config_file, args=[])
        except Exception as e:
            raise Exception(f"failed to parse {config_file}") from e
