import functools
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import jax.random as jrandom
import wandb

import haliax as hax
import haliax.random
from haliax.partitioning import named_jit

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data import ReplicatedBatchLoader, ShardedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LmExample
from levanter.logging import log_time_to_wandb
from levanter.lora import LoraConfig, combine_lora_params, loraize, partition_lora_params
from levanter.trainer import OptimizerConfig, Trainer, TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.py_utils import non_caching_cycle


logger = logging.getLogger(__name__)


@dataclass
class LoraLmConfig:
    initialize_from_hf: str
    lora: LoraConfig
    data: LMDatasetConfig = field(default_factory=LMDatasetConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None
    hf_save_steps: int = 1000

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

    # We have two axis_mappings: one for storing the model and optimizer states, and one for compute
    # This allows Zero-3-style parameter sharding, where we shard the parameters and optimizer state across the mesh
    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    eval_loader = ReplicatedBatchLoader(
        CausalLmDataset(config.data.token_seq_dataset("validation", Pos.size), Pos, KeyPos),
        config.trainer.device_mesh,
        EvalBatch,
        compute_axis_mapping,
    )

    train_loader = ShardedBatchLoader(
        CausalLmDataset(config.data.token_seq_dataset("train", Pos.size), Pos, KeyPos),
        config.trainer.device_mesh,
        Batch,
        compute_axis_mapping,
    )

    # We use Optax for our optimizer. It's a pretty standard library for optimizers in JAX.
    optimizer = config.optimizer.build(config.trainer.num_train_steps)

    with config.trainer.device_mesh:
        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        hf_model = converter.load_pretrained(model_config, axis_mapping=parameter_axis_mapping)

        # A note on the difference between "adapter_model" and "base_model":
        # In LoRA and other so-called "parameter-efficient fine-tuning" methods, we have two sets of parameters:
        # 1) The "base" parameters, which are the parameters of the original foundation model
        # 2) The "adapter" parameters, which are the parameters of the "head" of the model that we're fine-tuning

        # Structurally, Equinox works best if we keep these two sets of parameters separate. As an example,
        # consider a simple model with two parameters, attention and mlp. That might look like:
        # model = Model(attention=Attention(proj_qkv=Linear), mlp=Mlp())
        # with LoRA, our model might look more like:
        # model = Model(attention=Attention(proj_qkv=LoraLinear(wrapped=orig_proj_qkv, lora=LoraLinear(...)), mlp=Mlp())
        # we keep this partitioned as two trees:
        # base_model = Model(attention=Attention(proj_qkv=LoraLinear(wrapped=orig_proj_qkv, lora=None), mlp=Mlp())
        # adapter_model = Model(attention=Attention(proj_qkv=LoraLinear(wrapped=None, lora=LoraLinear(...)), mlp=None)
        # and then we combine them at runtime:
        # model = combine_lora_params(base_model, lora_params=adapter_model)
        # which just grounds out into a call to equinox.combine

        # We unfortunately need to pass these two trees around more or less together, only really distinguishing
        # them for gradients, optimizer state, and storing checkpoints.
        # Additionally, the gradient api assumes we compute gradients with respect to the first argument,
        # so adapter_model has to be the first argument to train_loss.
        @haliax.named_jit(axis_resources=parameter_axis_mapping)
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        combined_model = loraize_hf_model(hf_model)

        # next, split model into base and adapter and create initial optimizer state
        base_model, adapter_model = partition_lora_params(combined_model)

        del hf_model
        del combined_model

        def compute_loss(base_model, adapter_model, example: LmExample, inference, key=None):
            model = combine_lora_params(base_model, lora_params=adapter_model)
            return model.compute_loss(example, inference=inference, key=key).scalar()

        base_model = config.trainer.mp.cast_to_compute(base_model)

        # Our trainer is a wrapper around the optimizer and compute_loss function that handles checkpointing and fsdp
        trainer = Trainer(config.trainer, optimizer, functools.partial(compute_loss, base_model))
        state = trainer.initial_state(
            training_key,
            model=adapter_model,
        )

        all_param_count = parameter_count(combine_lora_params(base_model, adapter_model))
        wandb.summary["parameter_count"] = all_param_count
        just_lora_params = parameter_count(adapter_model)
        wandb.summary["trainable_parameter_count"] = just_lora_params
        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count%.3}")

        # boilerplate hooks and such
        trainer.add_default_hooks(eval_loader)
        trainer.add_hook(callbacks.log_performance_stats(Pos.size, trainer.config.train_batch_size), every=1)
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, trainer.config.run_id)
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
            model = trainer.mp.cast_to_compute(model)
            logprobs = model.compute_loss(example, inference=True, key=None, reduction=None)
            # roll forward to get the loss for each predicted token
            logprobs = hax.roll(logprobs, 1, Pos)
            return logprobs.rearrange((EvalBatch, Pos)).array

        trainer.add_hook(
            callbacks.compute_and_visualize_log_probs(
                eval_loader,
                tokenizer,
                partial(compute_log_probs, base_model),
                os.path.join(config.trainer.run_dir, "log_probs"),
                max_docs=EvalBatch.size,
            ),
            every=config.trainer.steps_per_eval,
        )

        # data loader. may need to seek to the right place if we're resuming
        iter_data = non_caching_cycle(train_loader)

        if state.step > 0:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(state.step + 1), desc="seeking data for resume"):
                next(iter_data)

            # TODO: this new initial step logic is not right
            initial_step = state.step + 1
        else:
            initial_step = 0

        # finally, run the training loop
        for step in range(initial_step, config.trainer.num_train_steps):
            with log_time_to_wandb("throughput/loading_time", step=step):
                example = next(iter_data)

            info = trainer.train_step(state, example, inference=False)
            state = info.state

            with log_time_to_wandb("throughput/hook_time", step=step):
                trainer.run_hooks(info)

        last_step = info
        trainer.run_hooks(last_step, force=True)
        # checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
