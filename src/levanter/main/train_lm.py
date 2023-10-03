import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.random as jrandom
import wandb

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


@dataclass
class TrainLmConfig:
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=Gpt2Config)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    fcm_prob: float = 0.0  # forgetful context masking prob. recommended 0.15

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000


def main(config: TrainLmConfig):
    tokenizer = config.data.the_tokenizer

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.default_hf_checkpoint_converter
        if tokenizer.vocab != converter.tokenizer.vocab:
            logger.warning("The tokenizers appear to be different. You may want to check this.")

        if isinstance(config.initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if config.use_hf_model_config:
            # TODO: log diff of old and new config
            # NB: gross mutability
            config.model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(config.model, HFCompatConfig):
        converter = config.model.default_hf_checkpoint_converter
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    # initialize training config *after* we've done the hf stuff b/c we might have changed the model config
    config.trainer.initialize(config)

    # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
    # this makes deterministic training pretty easy
    seed = config.trainer.seed
    data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

    # some axes we need
    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = config.model.Pos
    KeyPos = config.model.KeyPos

    # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
    # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    def compute_loss(model: LmHeadModel, example: LmExample, key=None):
        return model.compute_loss(example, key=key).scalar()

    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # Our trainer is a wrapper around the optimizer and compute_loss function that handles checkpointing and fsdp
    trainer = Trainer(config.trainer, optimizer, compute_loss)

    eval_dataset = CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos)
    eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)

    train_dataset = CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos)
    train_loader = trainer.sharded_loader(train_dataset, Batch)

    with trainer.device_mesh:
        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if state.step == 0:
            # TODO: I don't love that we init the model twice, but it's not a big deal i think?
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # this is a bit gross, but we want to free up the memory from the model we just built
                state.model = None
                model = converter.load_pretrained(config.model, axis_mapping=parameter_axis_mapping)
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        wandb.summary["parameter_count"] = parameter_count(state.model)

        # boilerplate hooks and such
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size), every=1)
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.run_id)

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter, upload_to_hf=config.hf_upload or False),
                every=config.hf_save_steps,
            )

        # visualize log probs
        @named_jit(
            in_axis_resources=parameter_axis_mapping,
            axis_resources=compute_axis_mapping,
            out_axis_resources=compute_axis_mapping,
        )
        def compute_log_probs(model, example: LmExample):
            model = trainer.mp.cast_to_compute(model)
            logprobs = model.compute_loss(example, key=None, reduction=None)
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            return logprobs.rearrange((EvalBatch, Pos)).array

        # engine.add_hook(
        #     callbacks.compute_and_visualize_log_probs(
        #         eval_loader, tokenizer, compute_log_probs, os.path.join(config.trainer.run_dir, "log_probs")
        #     ),
        #     every=config.trainer.steps_per_eval,
        # )
        #
        # data loader. may need to seek to the right place if we're resuming
        iter_data = non_caching_cycle(train_loader)

        if state.step > 0:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(state.step + 1), desc="seeking data for resume"):
                next(iter_data)

        ## OK, actually run training!
        trainer.train(state, iter_data)
        # checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
