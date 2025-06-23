import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jmp
import matplotlib.pyplot as plt
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.nn import log_softmax
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.text import (
    LMMixtureDatasetConfig,
    SingleDatasetLMConfig,
    UrlSingleDatasetLMConfig,
)
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device


logger = logging.getLogger(__name__)


@dataclass
class EvalSlidingLmConfig:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: SingleDatasetLMConfig | LMMixtureDatasetConfig = field(default_factory=UrlSingleDatasetLMConfig)
    model: LmConfig = field(default_factory=Gpt2Config)

    split: str = "validation"
    max_batches: Optional[int] = None


def main(config: EvalSlidingLmConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    Pos = config.model.Pos

    cache = config.data.build_or_load_cache(config.split)
    if cache is None:
        raise ValueError(f"No dataset found for split {config.split}")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def _to_example(row):
        ids = row["input_ids"].tolist()
        src_len = int(row["sources_len"])
        if len(ids) > Pos.size:
            ids = ids[: Pos.size]
        else:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens = hax.named(np.array(ids, dtype=np.int32), Pos)
        return LmExample.from_prompt_and_completion(Pos, tokens, prompt_length=src_len)

    dataset = cache.map(_to_example)

    loader = DataLoader(
        dataset,
        batch_size=config.trainer.eval_batch_size,
        axis_resources=config.trainer.compute_axis_mapping,
        mesh=config.trainer.device_mesh,
    )

    if config.max_batches is not None:
        loader = itertools.islice(loader, config.max_batches)

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        def compute_log_probs(model: LmHeadModel, batch: LmExample):
            model = mp.cast_to_compute(model)
            with hax.axis_mapping(compute_axis_mapping):
                logits = model(batch.tokens, attn_mask=batch.attn_mask)
                lp = log_softmax(logits, axis=model.Vocab)
                targets = hax.roll(batch.tokens, -1, Pos)
                lp = hax.take(lp, model.Vocab, targets)
                mask = 1 - hax.nn.one_hot(-1, Pos, dtype=lp.dtype)
                if batch.loss_mask is not None:
                    mask = mask * batch.loss_mask
                return lp * mask

        compute_log_probs = hax.named_jit(compute_log_probs, out_axis_resources=None)

        if config.checkpoint_path is not None:
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
        elif config.hf_checkpoint is not None:
            model_config = config.model
            if not hasattr(model_config, "hf_checkpoint_converter"):
                raise ValueError("Model config does not have an HF checkpoint converter.")
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=tokenizer)
            model = converter.load_pretrained(
                model_config.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype
            )
        else:
            raise ValueError("Must specify checkpoint_path or hf_checkpoint")

        log_probs = []
        for batch in loader:
            lp = compute_log_probs(model, batch)
            log_probs.append(np.array(lp))

        if not log_probs:
            raise ValueError("No data processed")

        lp_matrix = np.concatenate(log_probs, axis=0)
        prob_matrix = np.exp(lp_matrix)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(prob_matrix.T, vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_xlabel("Example")
        ax.set_ylabel("Position")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        path = "sliding_eval_heatmap.png"
        fig.savefig(path)
        levanter.tracker.log_artifact(path, name=path, type="plot")

    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
