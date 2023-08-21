import functools
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.random as jrandom
import jmp
import wandb

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.data import ReplicatedBatchLoader, ShardedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig
from levanter.logging import capture_time, log_time_to_wandb
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import OptimizerConfig, StepInfo, Trainer, TrainerConfig
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

    eval_loader = ReplicatedBatchLoader(
        CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos),
        config.trainer.device_mesh,
        EvalBatch,
        compute_axis_mapping,
        # max_capacity=None,
    )

    train_loader = ShardedBatchLoader(
        CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos),
        # TokenSeqDataset(config.data.build_or_load_cache("train"), Pos),
        config.trainer.device_mesh,
        Batch,
        compute_axis_mapping,
    )

    with config.trainer.device_mesh:
        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # Mixed Precision. See our tutorial at https://colab.research.google.com/drive/1_4cikwt-UhSH7yRzNRK8ze9msM9r2mEl
        mp: jmp.Policy = config.trainer.mp

        def compute_loss(model: LmHeadModel, example: LmExample, inference, key=None):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)
                return model.compute_loss(example, inference=inference, key=key).scalar()

        # eval loss needs to specify the parameter sharding
        eval_loss = functools.partial(
            named_jit(compute_loss, in_axis_resources=parameter_axis_mapping), inference=True
        )

        # We use Optax for our optimizer. It's a pretty standard library for optimizers in JAX.
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        # Our trainer is a wrapper around the optimizer and compute_loss function that handles checkpointing and fsdp
        trainer: Trainer[LmHeadModel, LmExample] = Trainer(config.trainer, optimizer, compute_loss)

        model, opt_state, training_key, resume_step = trainer.initial_state(
            lambda model_key: config.model.build(Vocab, key=model_key),
            training_key,
        )

        if resume_step is None:
            # no checkpoint was found, so we need to initialize the model and opt state
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # TODO: I don't love that we init the model twice, but it's not a big deal i think?
                del model
                model = converter.load_pretrained(config.model, axis_mapping=parameter_axis_mapping)
                model = named_jit(mp.cast_to_param, parameter_axis_mapping)(model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        wandb.summary["parameter_count"] = parameter_count(model)

        # boilerplate hooks and such
        trainer.add_hook(callbacks.pbar_logger(total=config.trainer.num_train_steps), every=1)
        trainer.add_hook(callbacks.log_to_wandb, every=1)
        trainer.add_hook(callbacks.log_performance_stats(Pos.size, config.trainer.train_batch_size), every=1)
        if config.trainer.max_eval_batches is None or config.trainer.max_eval_batches > 0:
            trainer.add_hook(
                callbacks.compute_validation_loss(eval_loss, eval_loader, max_batches=config.trainer.max_eval_batches),
                every=config.trainer.steps_per_eval,
            )
        trainer.add_hook(callbacks.wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = config.trainer.checkpointer.create(config.trainer.run_id)
        trainer.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, config.trainer.run_id)
            from levanter.compat.hf_checkpoints import save_hf_checkpoint_callback

            trainer.add_hook(
                save_hf_checkpoint_callback(full_save_path, converter),
                every=config.hf_save_steps,
            )

        # visualize log probs
        @named_jit(
            in_axis_resources=parameter_axis_mapping,
            axis_resources=compute_axis_mapping,
            out_axis_resources=compute_axis_mapping,
        )
        def compute_log_probs(model, example: LmExample):
            model = mp.cast_to_compute(model)
            logprobs = model.compute_loss(example, inference=True, key=None, reduction=None)
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

        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(resume_step + 1), desc="seeking data for resume"):
                next(iter_data)
            initial_step = resume_step + 1
        else:
            initial_step = 0

        # assign these here in case num_train_steps == 0
        step_loss = 0.0
        step_time = lambda: 0.0  # noqa: E731

        # finally, run the training loop
        for step in range(initial_step, config.trainer.num_train_steps):
            with capture_time() as step_time:
                with log_time_to_wandb("throughput/loading_time", step=step):
                    example = next(iter_data)
                    my_key, training_key = jrandom.split(training_key, 2)

                jax_step_loss, model, opt_state = trainer.train_step(
                    model, opt_state, example, key=my_key, inference=False
                )
                step_loss = jax_step_loss.item()  # type: ignore

            with log_time_to_wandb("throughput/hook_time", step=step):
                trainer.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, step_duration=step_time()))

        last_step = StepInfo(
            config.trainer.num_train_steps,
            model,
            opt_state,
            step_loss,
            training_key,
            step_duration=step_time(),
        )

        trainer.run_hooks(last_step, force=True)
        checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
