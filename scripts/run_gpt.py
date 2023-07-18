import functools
import logging
import time
from dataclasses import dataclass

import jax
import jmp
from tqdm import tqdm

import haliax as hax
from haliax import Axis
from haliax.nn import cross_entropy_loss
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter.data import ReplicatedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LmExample
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig

from jax_automin.interpreter import automin_function


logger = logging.getLogger(__name__)


@dataclass
class EvalLmConfig:
    trainer: TrainerConfig = TrainerConfig()
    data: LMDatasetConfig = LMDatasetConfig()
    model: LmConfig = Gpt2Config()

    eval_on_train: bool = False


def main(config: EvalLmConfig):
    config.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    Batch = Axis("batch", config.trainer.eval_batch_size)
    Pos = config.model.Pos
    KeyPos = config.model.Pos

    if config.eval_on_train:
        raw_dataset = CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos)
    else:
        raw_dataset = CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos)

    eval_loader = ReplicatedBatchLoader(raw_dataset, config.trainer.device_mesh, Batch)
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        def compute_loss(model: LmHeadModel, example: LmExample, key, inference):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(example.tokens, example.attn_mask, key=key, inference=inference)
                pred_y = mp.cast_to_output(pred_y)

                target_y = hax.nn.one_hot(example.targets, Vocab, dtype=pred_y.dtype)

                per_ex_loss = cross_entropy_loss(pred_y, Vocab, target_y, where=example.loss_mask, reduction_axis=Pos)
                return hax.mean(per_ex_loss).scalar()

        compute_loss_pjit = named_jit(compute_loss, axis_resources=parameter_axis_mapping)

        total = config.trainer.max_eval_batches

        # initialize the model
        model = config.model.build(Vocab, key=key)
        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

        total_loss = 0.0
        total_time = 0.0
        n = 0

        pbar = tqdm(eval_loader, desc="eval", position=1, leave=False)
        print(len(list(eval_loader)))
        for batch in pbar:
            print(n)
            # if n == 0:
            #     source = automin_function(functools.partial(compute_loss, inference=False), model, batch, key)
            #     print(source)
            this_key, key = jax.random.split(key)
            time_in = time.time()
            loss = compute_loss_pjit(model, batch, this_key, inference=False)
            total_loss += loss.item()
            time_out = time.time()
            if n > 0:
                total_time += time_out - time_in
            n += 1
            pbar.set_postfix(loss=total_loss / n)
            pbar.set_description(f"eval ({time_out - time_in:.3f}s)")

            if total and n >= total:
                break

        logger.info(f"eval loss: {total_loss / n:.3f}")
        logger.info(f"eval time: {total_time / (n-1):.3f}")


if __name__ == "__main__":
    levanter.config.main(main)()
