import logging
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jmp

import haliax as hax
from haliax import Axis
from haliax.partitioning import fsdp, round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.data import DataLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel, compute_next_token_loss
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode
from levanter.visualization import compute_and_visualize_log_probs


logger = logging.getLogger(__name__)


@dataclass
class VizGpt2Config:
    checkpoint_path: str
    path: str = "logprobs.html"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    model: LmConfig = field(default_factory=Gpt2Config)

    num_docs: int = 256


def main(config: VizGpt2Config):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    # some axes we use outside the model proper
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    eval_loader = DataLoader(
        EvalBatch,
        CausalLmDataset(config.data.validation_set(Pos.size), Pos, KeyPos),  # type: ignore
        32,
        config.trainer.device_mesh,
        config.trainer.compute_axis_mapping,
    )

    # some axes we use outside the model proper
    Pos = config.model.Pos

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

        @fsdp(parameter_axis_mapping, compute_axis_mapping)
        def compute_log_probs(model: LmHeadModel, example: LmExample):
            model = inference_mode(model, True)
            model = mp.cast_to_compute(model)
            logprobs = compute_next_token_loss(model, example, reduction=None)
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            return logprobs.rearrange((EvalBatch, Pos)).array

        # initialize the model
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")

        assert model is not None

        model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

        compute_and_visualize_log_probs(
            path=config.path,
            model=model,
            tokenizer=tokenizer,
            log_prob_fn=compute_log_probs,
            test_data=eval_loader,
            max_docs=config.num_docs,
        )


if __name__ == "__main__":
    levanter.config.main(main)()
