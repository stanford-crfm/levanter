import os
import tempfile

import jax
import pytest

import levanter.main.train_lm as train_lm
from levanter.logging import WandbConfig


curdir = os.getcwd()


@pytest.mark.entry
def test_train_lm():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as f:
        try:
            os.chdir(f)

            config = train_lm.TrainLmConfig(
                data=train_lm.LMDatasetConfig(id="dlwh/wikitext_103_detokenized", cache_dir=f"{curdir}/test_cache"),
                model=train_lm.Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    seq_len=128,
                    hidden_dim=32,
                ),
                trainer=train_lm.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                ),
            )
            train_lm.main(config)
        finally:
            os.chdir(curdir)
