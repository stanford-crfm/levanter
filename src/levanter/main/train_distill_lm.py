import dataclasses
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.random as jrandom
import jax.numpy as jnp

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning, ResourceMapping

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LMMixtureDatasetConfig
from levanter.models.factorized_llama import FactorizedLlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.layerwise_trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count


logger = logging.getLogger(__name__)


@dataclass
class TrainDistillLmConfig:
    data: Union[LMDatasetConfig, LMMixtureDatasetConfig] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    teacher: LmConfig = field(default_factory=FactorizedLlamaConfig)
    student: LmConfig = field(default_factory=FactorizedLlamaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15

    init_from_hf: bool = False
    hf_save_path: Optional[str] = None
    update_hessian_steps: int = 10


def main(config: TrainDistillLmConfig):
    tokenizer = config.data.the_tokenizer

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer) as trainer:
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, student_key, teacher_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 5)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        print("Parameters", parameter_axis_mapping)
        print("Compute", compute_axis_mapping)

        # some axes we need
        Batch = config.trainer.TrainBatch
        EvalBatch = config.trainer.EvalBatch
        Pos = config.teacher.Pos
        KeyPos = config.teacher.KeyPos

        tagged_eval_datasets = config.data.tagged_eval_sets(Pos.size)
        train_dataset = CausalLmDataset(
            config.data.train_set(Pos.size), Pos, KeyPos, ignore_index=config.data.ignore_token_id
        )

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        def _load_model_from_hf(model_config: LmConfig):
            # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
            # I recommend skipping it for now
            assert isinstance(model_config, HFCompatConfig)
            converter = model_config.default_hf_checkpoint_converter
            if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
                logger.warning("The tokenizers appear to be different. You may want to check this.")

            converter = converter.replaced(tokenizer=tokenizer)
            # initialize from an hf pretrained model
            logger.info(f"Initializing model from HF checkpoint '{converter.reference_checkpoint}'")
            model = converter.load_pretrained(
                model_config, axis_mapping=parameter_axis_mapping, dtype=trainer.mp.compute_dtype
            )
            logger.info("Initialized model, casting.")
            model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
            logger.info("Ready.")
            return model

        state = trainer.initial_state(
            training_key,
            student_init=lambda: config.student.build(Vocab, key=student_key),
            teacher_init=lambda: config.teacher.build(Vocab, key=teacher_key),
        )

        if int(state.step) == 0 and config.init_from_hf:
            state = dataclasses.replace(state, teacher=None)
            gc.collect()
            teacher = _load_model_from_hf(config.teacher)
            gc.collect()
            # student = _load_model_from_hf(config.student)
            # gc.collect()
            state = dataclasses.replace(state, teacher=teacher)

        levanter.tracker.log_summary(
            {
                "teacher_parameter_count": parameter_count(state.teacher),
                "student_parameter_count": parameter_count(state.student),
            }
        )

        train_loader = iter(trainer.sharded_loader(train_dataset, Batch))

        if int(state.step) > 0:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(state.step), desc="seeking data for resume"):
                next(train_loader)

        ## OK, actually run training!
        trainer.train(state, train_loader)
        # checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
