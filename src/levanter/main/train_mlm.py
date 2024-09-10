# train_mlm.py

import dataclasses
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union

import jax
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCompatConfig, save_hf_checkpoint_callback
from levanter.data.text import MaskedLmDataset, LMDatasetConfig, LMMixtureDatasetConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.models.roberta import RobertaConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count

logger = logging.getLogger(__name__)

@dataclass
class TrainMlmConfig:
    data: Union[LMDatasetConfig, LMMixtureDatasetConfig] = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=RobertaConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    # config related to continued pretraining
    initialize_from_hf: Union[bool, str] = False
    """if provided, this will override the model config in the config. if true, use the default hf checkpoint for this model class"""
    use_hf_model_config: bool = False  # if true, replace the model config with the hf config from the checkpoint

    # TODO: atm we don't support loading from a checkpoint that has a different tokenizer. this is a bit annoying
    # TODO: atm you have to at least specify a levanter model config with the same type as the hf checkpoint

    mlm_prob: float = 0.15  # masking probability for MLM
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 10000

    update_hessian_steps: int = 10
    data_seed: Optional[int] = None  # if provided, will override the data seed from the trainer

def main(config: TrainMlmConfig):
    tokenizer = config.data.the_tokenizer

    # this is some unpleasant code to allow us to initialize from a hf checkpoint. If this is your first read through,
    # I recommend skipping it for now
    if config.initialize_from_hf:
        if config.trainer.initialize_from is not None:
            raise ValueError("Cannot specify both initialize_from_hf and initialize_from")

        assert isinstance(config.model, HFCompatConfig)
        converter = config.model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
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
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    levanter.initialize(config)
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    # Using the trainer as a context manager does 3 things:
    # 1. Sets the device mesh
    # 2. Sets the axis mapping (for fsdp)
    # 3. Sets the global metrics tracker
    with Trainer(config.trainer, optimizer) as trainer, jax.disable_jit(True):
        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        if config.data_seed is not None:
            logger.info(f"Overriding data seed with {config.data_seed}")
            data_key = jrandom.PRNGKey(config.data_seed)

        # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
        # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
        compute_axis_mapping = trainer.compute_axis_mapping
        parameter_axis_mapping = trainer.parameter_axis_mapping

        # some axes we need
        Batch = config.trainer.TrainBatch
        EvalBatch = config.trainer.EvalBatch
        Pos = config.model.Pos
        KeyPos = config.model.KeyPos

        tagged_eval_datasets = config.data.tagged_eval_sets(Pos.size)
        mask_id = tokenizer.mask_token_id
        train_dataset = MaskedLmDataset(
            config.data.train_set(Pos.size, key=data_key), Pos, KeyPos,
            mask_token_id=mask_id,
            mask_prob=config.mlm_prob, key=data_key, #ignore_index=config.data.ignore_token_id
        )

        # to do partitioning, our dimensions have to be divisible by the size of the physical axes they're mapped to
        # For most things, we just insist you specify the config right, but tokenizers often have strange numbers of
        # tokens: gpt-2 has 50257, for example. So we round up.
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        state = trainer.initial_state(training_key, model_init=lambda: config.model.build(Vocab, key=model_key))

        if int(state.step) == 0:
            # TODO: I don't love that we init the model twice, but it's not a big deal i think?
            if config.initialize_from_hf:
                # initialize from an hf pretrained model
                logger.info(
                    "No training checkpoint found. Initializing model from HF checkpoint"
                    f" '{converter.reference_checkpoint}'"
                )
                # this is a bit gross, but we want to free up the memory from the model we just built
                state = dataclasses.replace(state, model=None)
                gc.collect()
                model = converter.load_pretrained(
                    config.model.model_type,
                    config.model,
                    axis_mapping=parameter_axis_mapping,
                    dtype=trainer.mp.compute_dtype,
                )
                model = named_jit(trainer.mp.cast_to_param, parameter_axis_mapping)(model)
                state = dataclasses.replace(state, model=model)
            else:
                logger.info("No checkpoint found. Starting from scratch.")

        levanter.tracker.log_summary({"parameter_count": parameter_count(state.model)})

        if len(tagged_eval_datasets) == 0:
            logger.warning("No evaluation datasets provided.")
        else:
            masked_datasets = [
                (MaskedLmDataset(ds, Pos, KeyPos, mask_prob=config.mlm_prob, key=data_key, mask_token_id=mask_id), tags)
                for ds, tags in tagged_eval_datasets
            ]
            max_eval_examples_per_ds = config.trainer.max_eval_batches
            if max_eval_examples_per_ds is not None:
                max_eval_examples_per_ds *= config.trainer.eval_batch_size

            cb = levanter.eval.cb_tagged_lm_evaluate(
                EvalBatch, masked_datasets, trainer.device_mesh, compute_axis_mapping, max_eval_examples_per_ds
            )
            trainer.add_hook(cb, every=config.trainer.steps_per_eval)

        flops_per_token = config.model.flops_per_token(vocab_size)
        flops_per_example = 3 * flops_per_token * Pos.size if flops_per_token is not None else None
        trainer.add_hook(
            callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size, flops_per_example), every=1
        )
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
        def compute_log_probs(model, example):
            model = trainer.mp.cast_to_compute(model)
            logprobs = model.compute_loss(example, key=None, reduction=None)
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            return logprobs.rearrange((EvalBatch, Pos)).array

        train_loader = iter(trainer.sharded_loader(train_dataset, Batch))

        if int(state.step) > 0:
            import tqdm
            for _ in tqdm.tqdm(range(state.step), desc="seeking data for resume"):
                next(train_loader)

        trainer.train(state, train_loader)

if __name__ == "__main__":
    levanter.config.main(main)()