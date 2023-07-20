import os
import tempfile

import jax
import pytest
import ray

import levanter.main.train_lm as train_lm
from levanter.distributed import RayConfig
from levanter.logging import WandbConfig


def setup_module(module):
    ray.init("local", num_cpus=10)


def teardown_module(module):
    ray.shutdown()


@pytest.mark.entry
def test_train_lm():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory():
        try:
            config = train_lm.TrainLmConfig(
                data=train_lm.LMDatasetConfig(id="dlwh/wikitext_103_detokenized", cache_dir="test_cache"),
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
                    ray=RayConfig(auto_start_cluster=False),
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
