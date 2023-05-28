import numpy as onp

from jax.random import PRNGKey

import haliax as hax

from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig
from levanter.models.backpack import BackpackConfig, BackpackLMHeadModel


VOCAB_SIZE = 50264


def test_backpack_predict():
    trainer_config = TrainerConfig()
    
    Vocab = round_axis_for_partitioning(Axis("vocab", VOCAB_SIZE), trainer_config.compute_axis_mapping)
    model_config = BackpackConfig()
    model_key = PRNGKey(0)
    model = BackpackLMHeadModel(Vocab, model_config, key=model_key)
    mp = trainer_config.mp
    model = mp.cast_to_param(model)
    
    input = hax.random.randint(PRNGKey(0), model.Pos, 0, model.Vocab.size)
    attn_mask = hax.nn.attention.causal_mask(model.Pos, model.config.KeyPos)

    def compute(input):
        return hax.nn.softmax(
            model(input, inference=True, key=None, attn_mask=attn_mask),
            axis=model.Vocab,
        )

    out_1 = compute(input).array
    assert out_1.shape == (
        model.Pos.size,
        model.Vocab.size,
    ), f"{out_1.shape} != {(model.Pos, model.Vocab.size)}"


