import os
import tempfile

import jax
import pytest

import levanter.main.train_asr as train_asr
import tiny_test_corpus
from levanter.distributed import RayConfig
from levanter.tracker.wandb import WandbConfig


@pytest.mark.entry
def test_train_asr():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.tiny_asr_corpus_config(tmpdir)
        try:
            config = train_asr.TrainASRConfig(
                data=data_config,
                model=train_lm.Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    seq_len=64,
                    d_model=32,
                    use_flash_attention=True,
                ),
                trainer=train_asr.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                    hf_save_path=f"{path}/hf_asr_output",
                ),
            )
            train_asr.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
