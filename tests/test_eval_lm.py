import os
import tempfile

import jax
import pytest

import haliax

import levanter.main.eval_lm as eval_lm
from levanter.checkpoint import save_checkpoint
from levanter.logging import WandbConfig
from levanter.models.gpt2 import Gpt2LMHeadModel


@pytest.mark.entry
def test_eval_lm():
    # just testing if eval_lm has a pulse
    # save a checkpoint
    model_config = eval_lm.Gpt2Config(
        num_layers=2,
        num_heads=2,
        seq_len=128,
        hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as f:
        try:
            data_config = eval_lm.LMDatasetConfig(id="dlwh/wikitext_103_detokenized", cache_dir="test_cache")
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
                ),
                checkpoint_path=f"{f}/ckpt",
            )
            eval_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
