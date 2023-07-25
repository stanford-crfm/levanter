import os
import tempfile

import jax
import pytest
import ray

import levanter.main.train_lm as train_lm
import tiny_test_corpus
from levanter.distributed import RayConfig
from levanter.logging import WandbConfig
from levanter.utils.py_utils import logical_cpu_core_count


def setup_module(module):
    ray_designated_cores = max(1, logical_cpu_core_count())
    ray.init("local", num_cpus=ray_designated_cores)


def teardown_module(module):
    ray.shutdown()


@pytest.mark.entry
def test_train_lm():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config = tiny_test_corpus.tiny_corpus_config(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=train_lm.Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    seq_len=32,
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
