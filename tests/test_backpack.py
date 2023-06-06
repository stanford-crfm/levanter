import glob
import os

from jax.random import PRNGKey

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
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
