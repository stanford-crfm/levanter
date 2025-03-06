import os
import tempfile

import jax
import pytest

from haliax.quantization import QuantizationConfig

import levanter.main.train_lm as train_lm
import tiny_test_corpus
from levanter.distributed import RayConfig
from levanter.tracker.wandb import WandbConfig


@pytest.mark.entry
def test_train_lm():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=train_lm.Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    seq_len=64,
                    hidden_dim=32,
                    attn_backend=None,  # use default for platform
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


@pytest.mark.entry
def test_train_lm_fp8():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=train_lm.Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    seq_len=64,
                    hidden_dim=32,
                    attn_backend=None,  # use default for platform
                ),
                trainer=train_lm.TrainerConfig(
                    quantization=QuantizationConfig(fp8=True),
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
