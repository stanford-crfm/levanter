import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import jax.random as jrandom
import numpy as np

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data import AsyncDataset
from levanter.data.dataset import T_co
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LMMixtureDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, compute_next_token_loss
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    data: Union[LMDatasetConfig, LMMixtureDatasetConfig] = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15
    z_loss_weight: float = 0.0

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000

    update_hessian_steps: int = 10
    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer
    initialize_from_checkpoint_path: Optional[str] = None
    # if provided, will initialize from this checkpoint, used for llama style data mixture


def main(config: TrainLmConfig):
    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    loss_function = functools.partial(compute_next_token_loss, logsumexp_weight=config.z_loss_weight)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer, loss_function) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        Batch = config.trainer.TrainBatch
        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos
        KeyPos = config.model.KeyPos

        class DummyArrayDataset(AsyncDataset[np.ndarray]):
            def __init__(self, length, seqlen, vocab_size):
                self.length = length
                self.rng = np.random.default_rng(seed=0)
                self.seqlen = seqlen
                self.vocab_size = vocab_size

            async def async_len(self) -> int:
                return self.length

            async def final_length_is_known(self) -> bool:
                return True

            def is_finite(self) -> bool:
                return True

            async def current_len(self) -> Optional[int]:
                return self.length

            async def get_batch(self, indices: Sequence[int]) -> Sequence[T_co]:
                return [self.rng.integers(0, self.vocab_size, size=(self.seqlen,), dtype=np.int32) for _ in indices]

        # 40 batches is roughly the length in the real dataset
        tagged_eval_datasets: list = [(DummyArrayDataset(EvalBatch.size * 40, Pos.size, 50257), ["dummy"])]
        # train_dataset = CausalLmDataset(
        #     config.data.train_set(Pos.size, key=data_key), Pos, KeyPos, ignore_index=config.data.ignore_token_id
        # )
        train_dataset = CausalLmDataset(DummyArrayDataset(100000000, Pos.size, 50257), Pos, KeyPos)

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = 50257
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        if len(tagged_eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            causal_datasets = [(CausalLmDataset(ds, Pos, KeyPos), tags) for ds, tags in tagged_eval_datasets]

            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch,
                causal_datasets,
                trainer.device_mesh,
                compute_axis_mapping,
                mp=config.trainer.mp,
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        train_loader = trainer.data_loader(train_dataset, Batch)
        train_loader = train_loader.iter_from_step(state.step)

        ## OK, actually run training!
        trainer.train(state, train_loader)
        # checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
