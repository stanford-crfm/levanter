import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import equinox as eqx
import jax.random as jrandom
import jmp
import wandb

import haliax as hax
import haliax.random
from haliax import Axis
from haliax.nn import cross_entropy_loss
from haliax.partitioning import named_jit, round_axis_for_partitioning

import levanter
from levanter import callbacks
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data import ReplicatedBatchLoader, ShardedBatchLoader
from levanter.data.text import CausalLmDataset, LMDatasetConfig, LmExample
from levanter.grad_accum import accumulate_gradients_sharded
from levanter.logging import capture_time, log_time_to_wandb
from levanter.lora import LoraConfig, combine_lora_params, loraize, partition_lora_params
from levanter.models.lm_model import LmHeadModel
from levanter.trainer import OptimizerConfig, StepInfo, TrainerConfig, TrainerHooks
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

    if isinstance(config.initialize_from_hf, str):
        converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
    else:
        converter = converter.replaced(tokenizer=tokenizer)

    config.trainer.initialize(config)
    model_config = converter.default_config

    # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
    # this makes deterministic training pretty easy
    seed = config.trainer.seed
    data_key, loader_key, model_key, lora_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 5)

    # some axes we need
    Batch = Axis("batch", config.trainer.train_batch_size)
    EvalBatch = Axis("batch", config.trainer.eval_batch_size)
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
        # TokenSeqDataset(config.data.build_or_load_cache("train"), Pos),
        config.trainer.device_mesh,
        Batch,
        compute_axis_mapping,
    )

    with config.trainer.device_mesh as mesh:
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), parameter_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp
        optimizer = config.optimizer.build(config.trainer.num_train_steps)

        def compute_loss(model: LmHeadModel, example: LmExample, key, inference):
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(example.tokens, example.attn_mask, key=key, inference=inference)
                pred_y = mp.cast_to_output(pred_y)

                target_y = hax.nn.one_hot(example.targets, Vocab, dtype=pred_y.dtype)

                return cross_entropy_loss(pred_y, Vocab, target_y, where=example.loss_mask, reduction_axis=Pos)

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

        @named_jit(axis_resources=parameter_axis_mapping)
        def train_loss(adapter_model, base_model, example, key):
            model = combine_lora_params(base_model, lora_params=adapter_model)
            return hax.mean(compute_loss(model, example, key, inference=False)).scalar()

        @named_jit(axis_resources=parameter_axis_mapping, donate_args=(False, True, True, False, False))
        def train_step(base_model, adapter_model, opt_state, examples: LmExample, key):
            grad_loss = eqx.filter_value_and_grad(train_loss)

            loss, grads = accumulate_gradients_sharded(
                grad_loss,
                Batch,
                adapter_model,
                base_model,  # base_model is an "input" b/c we don't want to compute its gradient
                examples,
                key=key,
                per_device_parallelism=config.trainer.per_device_parallelism,
                parameter_axis_mapping=parameter_axis_mapping,
            )

            # distribute gradients across the mesh and apply them
            updates, opt_state = optimizer.update(grads, opt_state, params=adapter_model)
            adapter_model = eqx.apply_updates(adapter_model, updates)

            return loss, adapter_model, opt_state

        @named_jit(axis_resources=parameter_axis_mapping)
        def eval_loss(base_model, adapter_model, example):
            model = combine_lora_params(base_model, lora_params=adapter_model)
            return hax.mean(compute_loss(model, example, None, True)).scalar()

        # load the underlying hf model
        logger.info(f"Loading pretrained model from {converter.reference_checkpoint}")
        hf_model = converter.load_pretrained(model_config, axis_mapping=parameter_axis_mapping)

        @haliax.named_jit(axis_resources=parameter_axis_mapping)
        def loraize_hf_model(model):
            return loraize(model, config.lora, key=lora_key)

        combined_model = loraize_hf_model(hf_model)

        # next, split model into base and adapter and create initial optimizer state
        base_model, adapter_model = partition_lora_params(combined_model)
        opt_state = haliax.named_jit(optimizer.init, axis_resources=parameter_axis_mapping)(adapter_model)

        del hf_model
        del combined_model

        # next, try to load the adapter model and opt state from a checkpoint. This may throw if we required a
        # checkpoint but it wasn't found.
        adapter_model, (opt_state, training_key), resume_step = config.trainer.maybe_load_checkpoint(
            adapter_model,
            (opt_state, training_key),
            axis_mapping=parameter_axis_mapping,
            mesh=mesh,
        )

        all_param_count = parameter_count(combine_lora_params(base_model, adapter_model))
        wandb.summary["parameter_count"] = all_param_count
        just_lora_params = parameter_count(adapter_model)
        wandb.summary["trainable_parameter_count"] = just_lora_params
        logger.info(f"Total parameter count: {all_param_count}")
        logger.info(f"Trainable parameter count: {just_lora_params}")
        logger.info(f"Fraction of parameters that are trainable: {just_lora_params * 1.0 / all_param_count%.3}")

        # boilerplate hooks and such
        engine = TrainerHooks()
        engine.add_hook(callbacks.pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(callbacks.log_to_wandb, every=1)
        engine.add_hook(callbacks.log_performance_stats(Pos.size, config.trainer.train_batch_size), every=1)
        engine.add_hook(
            callbacks.compute_validation_loss(
                partial(eval_loss, base_model), eval_loader, max_batches=config.trainer.max_eval_batches
            ),
            every=config.trainer.steps_per_eval,
        )
        engine.add_hook(callbacks.wandb_xla_logger(config.trainer.wandb), every=config.trainer.steps_per_eval)
        # engine.add_hook(callbacks.log_memory_usage(), every=1)
        checkpointer = config.trainer.checkpointer.create(config.trainer.run_id)
        engine.add_hook(checkpointer.on_step, every=1)  # checkpointer manages its own frequency
        if config.hf_save_path is not None:
            full_save_path = os.path.join(config.hf_save_path, config.trainer.run_id)
            from levanter.lora import save_peft_checkpoint_callback

            engine.add_hook(
                save_peft_checkpoint_callback(
                    full_save_path, config.lora, config.initialize_from_hf, upload_to_hf=config.hf_upload
                ),
                every=config.hf_save_steps,
            )

        # visualize log probs
        @named_jit(axis_resources=parameter_axis_mapping)
        def compute_log_probs(base_model, adapter_model, example: LmExample):
            """This method differs from eval_loss in that it skips the mean call, so we get a loss for each token"""
            model = combine_lora_params(base_model, adapter_model)
            with hax.axis_mapping(compute_axis_mapping):
                model = mp.cast_to_compute(model)

                pred_y = model(example.tokens, example.attn_mask, inference=True, key=None)
                pred_y = mp.cast_to_output(pred_y)
                targets = hax.nn.one_hot(example.tokens, Vocab, dtype=pred_y.dtype)
                loss = cross_entropy_loss(pred_y, Vocab, targets, where=example.loss_mask, reduction=None)
                logprobs = -loss
                # roll forward to get the loss for each predicted token
                logprobs = haliax.roll(logprobs, 1, Pos)
                return logprobs.rearrange((EvalBatch, Pos)).array

        engine.add_hook(
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

        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(resume_step + 1), desc="seeking data for resume", leave=False, offset=1):
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

                jax_step_loss, adapter_model, opt_state = train_step(
                    base_model, adapter_model, opt_state, example, my_key
                )
                step_loss = jax_step_loss.item()

            with log_time_to_wandb("throughput/hook_time", step=step):
                engine.run_hooks(
                    StepInfo(step, adapter_model, opt_state, step_loss, training_key, step_duration=step_time())
                )

        last_step = StepInfo(
            config.trainer.num_train_steps,
            adapter_model,
            opt_state,
            step_loss,
            training_key,
            step_duration=step_time(),
        )

        engine.run_hooks(last_step, force=True)
        checkpointer.on_step(last_step, force=True)


if __name__ == "__main__":
    levanter.config.main(main)()
