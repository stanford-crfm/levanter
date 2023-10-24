import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import jax.random as jrandom
import wandb

import haliax.random

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LmExample
from levanter.lora import (
    LoraConfig,
    lora_trainable_params_filter,
    loraize,
    save_merged_hf_checkpoint_callback,
    save_peft_checkpoint_callback,
)
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


@dataclass
class LoraLmConfig:
    initialize_from_hf: str
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    peft_save_path: Optional[str] = None  # path to save peft-compatible checkpoints
    peft_hf_upload: Optional[str] = None
    hf_save_steps: int = 1000

    merged_hf_save_path: Optional[str] = None  # path to save merged hf checkpoints
    merged_hf_upload: Optional[str] = None

    trust_remote_code: bool = False


def main(config: LoraLmConfig):
    tokenizer = config.data.the_tokenizer

    converter = HFCheckpointConverter.from_hf(config.initialize_from_hf, trust_remote_code=config.trust_remote_code)
    if tokenizer.vocab != converter.tokenizer.vocab:
        logger.warning("The tokenizers appear to be different. You may want to check this.")

    converter = converter.replaced(tokenizer=tokenizer)

    config.trainer.initialize(config)
    model_config = converter.default_config

    # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
    # this makes deterministic training pretty easy
    seed = config.trainer.seed
    data_key, loader_key, model_key, lora_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 5)

    # some axes we need
    Batch = config.trainer.TrainBatch
    EvalBatch = config.trainer.EvalBatch
    Pos = model_config.Pos
    KeyPos = model_config.KeyPos

    with config.trainer.device_mesh:
        # how we shard parameters across devices
        parameter_axis_mapping = config.trainer.parameter_axis_mapping

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        model = converter.load_pretrained(model_config, axis_mapping=parameter_axis_mapping)

        @haliax.named_jit(axis_resources=parameter_axis_mapping, donate_args=(True))
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        model = loraize_hf_model(model)

        lora_param_filter = lora_trainable_params_filter(model)

        def compute_loss(model, example: LmExample, key=None):
            return model.compute_loss(example, key=key).scalar()

        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        # Our trainer is a wrapper around the optimizer and compute_loss function that handles checkpointing and fsdp
        trainer = Trainer(config.trainer, optimizer, compute_loss, is_trainable=lora_param_filter)
        state = trainer.initial_state(training_key, model=model)

        all_param_count = parameter_count(state.model)
        just_lora_params = parameter_count(trainer.trainable_params_only(state.model))

        wandb.summary["parameter_count"] = all_param_count
        wandb.summary["trainable_parameter_count"] = just_lora_params
        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count%.3}")

        # data loaders
        eval_dataset = CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos)
        eval_loader = trainer.replicated_loader(eval_dataset, EvalBatch)

        train_dataset = CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos)
        train_loader = trainer.sharded_loader(train_dataset, Batch)

        # boilerplate hooks and such
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size), every=1)
        if config.peft_save_path is not None:
            full_save_path = os.path.join(config.peft_save_path, trainer.run_id)
            trainer.add_hook(
                save_peft_checkpoint_callback(
                    full_save_path, config.lora, config.initialize_from_hf, tokenizer, config.peft_hf_upload
                ),
                every=config.hf_save_steps,
            )

        if config.merged_hf_save_path is not None:
            full_save_path = os.path.join(config.merged_hf_save_path, trainer.run_id)
            trainer.add_hook(
                save_merged_hf_checkpoint_callback(full_save_path, converter, config.merged_hf_upload),
                every=config.hf_save_steps,
            )

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


if __name__ == "__main__":
    levanter.config.main(main)()
