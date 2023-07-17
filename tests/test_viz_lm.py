import os
import tempfile

import jax
import pytest

import haliax

import levanter.main.viz_logprobs as viz_logprobs
from levanter.checkpoint import save_checkpoint
from levanter.logging import WandbConfig
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


@pytest.mark.entry
def test_viz_lm():
    # just testing if eval_lm has a pulse
    # save a checkpoint
    model_config = Gpt2Config(
        num_layers=2,
        num_heads=2,
        seq_len=128,
        hidden_dim=32,
    )

    curdir = os.getcwd()
    with tempfile.TemporaryDirectory() as f:
        data_config = viz_logprobs.LMDatasetConfig(
            id="dlwh/wikitext_103_detokenized", cache_dir=f"{curdir}/test_cache"
        )
        tok = data_config.the_tokenizer
        Vocab = haliax.Axis("vocab", len(tok))
        model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

        save_checkpoint(model, None, 0, f"{f}/ckpt")

        config = viz_logprobs.VizGpt2Config(
            data=data_config,
            model=model_config,
            trainer=viz_logprobs.TrainerConfig(
                per_device_eval_parallelism=len(jax.devices()),
                max_eval_batches=1,
                wandb=WandbConfig(mode="disabled"),
                require_accelerator=False,
            ),
            checkpoint_path=f"{f}/ckpt",
            num_docs=len(jax.devices()),
            output_dir=f"{f}/viz",
        )
        viz_logprobs.main(config)
