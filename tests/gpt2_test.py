import dataclasses
import glob
import os

import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


def test_gradient_checkpointing():
    # ensure that gradient checkpointing doesn't change the output
    # (this is a regression test for a bug that caused the output to change)
    for num_blocks in [1, 2, 4, 8, 12]:
        config = Gpt2Config(
            seq_len=16,
            hidden_dim=72,
            num_layers=num_blocks,
            num_heads=8,
            gradient_checkpointing=False,
        )
        config_checkpoint = dataclasses.replace(config, gradient_checkpointing=True)
        key = PRNGKey(0)

        Vocab = Axis("vocab", 128)

        model = Gpt2LMHeadModel.init(Vocab, config, key=key)
        model_checkpoint = Gpt2LMHeadModel.init(Vocab, config_checkpoint, key=key)

        input_ids = hax.arange(config.Pos, dtype=jnp.int32)

        causal_mask = hax.nn.attention.causal_mask(config.Pos, config.KeyPos)

        a1 = model(input_ids, inference=False, key=key, attn_mask=causal_mask)
        a2 = model_checkpoint(input_ids, inference=False, key=key, attn_mask=causal_mask)

        assert hax.all(hax.isclose(a1, a2, rtol=1e-4, atol=1e-5)), f"failed with num_blocks={num_blocks}"


def test_gpt2_configs():
    # load the TrainGpt2Config from ../examples/gpt2_example.py
    test_path = os.path.dirname(os.path.abspath(__file__))
    gpt2_configs = os.path.join(test_path, "..", "config")

    # load module. might not be in pythonpath so we have to do this
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gpt2_example", os.path.join(test_path, "..", "examples", "gpt2_example.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TrainerConfig = module.TrainLmConfig

    for config_file in glob.glob(os.path.join(gpt2_configs, "gpt2_*.yaml")):
        try:
            import pyrallis

            pyrallis.parse(TrainerConfig, config_file, args=[])
        except Exception as e:
            raise Exception(f"failed to parse {config_file}") from e
