import logging
from dataclasses import dataclass

import equinox as eqx
import jax
import jmp

import haliax as hax
from haliax import Axis
from haliax.partitioning import fsdp, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.data import ReplicatedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import StepInfo, TrainerConfig


logger = logging.getLogger(__name__)


@dataclass
class VizGpt2Config:
    checkpoint_path: str
    output_dir: str = "logprob_viz"
    trainer: TrainerConfig = TrainerConfig()
    data: LMDatasetConfig = LMDatasetConfig()
    model: LmConfig = Gpt2Config()

    num_docs: int = 256


def main(config: VizGpt2Config):
    config.trainer.initialize(config)
    tokenizer = config.data.the_tokenizer

    EvalBatch = Axis("batch", config.trainer.eval_batch_size)

    # some axes we use outside the model proper
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    eval_loader = ReplicatedBatchLoader(
        CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos),
        config.trainer.device_mesh,
        EvalBatch,
    )

    # some axes we use outside the model proper
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        # don't want to compute the mask w.r.t. the final token

        @fsdp(parameter_axis_mapping, compute_axis_mapping, mp)
        def compute_log_probs(model: LmHeadModel, example: LmExample):
            logprobs = model.compute_loss(example, inference=True, key=None, reduction=None)
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            return logprobs.rearrange((EvalBatch, Pos)).array

        # initialize the model
        with jax.default_device(jax.devices("cpu")[0]):
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
            ckpt = load_checkpoint(model, None, config.checkpoint_path)

        assert ckpt is not None
        model, _, _ = ckpt

        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

        cb = callbacks.compute_and_visualize_log_probs(
            eval_loader, tokenizer, compute_log_probs, config.output_dir, max_docs=config.num_docs
        )
        cb(StepInfo(model=model, step=0, opt_state=None, loss=0.0, step_duration=0.0, next_key=0.0))

        del model


if __name__ == "__main__":
    levanter.config.main(main)()
