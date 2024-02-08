import os
import tempfile

import jax
import pytest
import ray

import haliax

import levanter.main.eval_lm as eval_lm
import tiny_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.distributed import RayConfig
from levanter.logging import WandbConfig
from levanter.models.gpt2 import Gpt2LMHeadModel
from levanter.utils.py_utils import logical_cpu_core_count


def setup_module(module):
    ray_designated_cores = max(1, logical_cpu_core_count())
    ray.init("local", num_cpus=ray_designated_cores)


def teardown_module(module):
    ray.shutdown()


@pytest.mark.entry
def test_eval_lm():
    # just testing if eval_lm has a pulse
    # save a checkpoint
    model_config = eval_lm.Gpt2Config(
        num_layers=2,
        num_heads=2,
        seq_len=32,
        hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as f:
        try:
            data_config = tiny_test_corpus.tiny_corpus_config(f)
            tok = data_config.the_tokenizer
            Vocab = haliax.Axis("vocab", len(tok))
            model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

            save_checkpoint(model, None, 0, f"{f}/ckpt")

            config = eval_lm.EvalLmConfig(
                data=data_config,
                model=model_config,
                trainer=eval_lm.TrainerConfig(
                    per_device_eval_parallelism=len(jax.devices()),
                    max_eval_batches=1,
                    wandb=WandbConfig(mode="disabled"),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
                checkpoint_path=f"{f}/ckpt",
            )
            eval_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
