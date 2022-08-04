import itertools
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
from jax import vmap
from jax.experimental.pjit import pjit
from jax.interpreters.pxla import PartitionSpec

import levanter.jax_utils
from haliax import Axis
from levanter import callbacks
from levanter.axis_names import ResourceAxis, infer_resource_partitions
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.logging import log_performance_stats
from levanter.logging import pbar_logger, log_to_wandb
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel

print(Counter([type(dev) for dev in jax.devices()]))
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import optax
import pyrallis
import wandb
from transformers import GPT2Tokenizer

from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig, WandbConfig
from levanter.jax_utils import shaped_rng_split, parameter_count
from levanter.modeling_utils import accumulate_gradients
from levanter.trainer_hooks import TrainerHooks, StepInfo


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()

    run_base_dir: str = "runs/"
    checkpoint_dir: str = "checkpoints/"

    dtype: jnp.dtype = jnp.float32


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.wandb.init(config)
    run_name = wandb.run.name or wandb.run.id
    run_dir = f"{config.run_base_dir}/{run_name}"
    checkpoint_dir = f"{config.checkpoint_dir}/{run_name}"

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer
    dataset = ShardedIndexedDataset(config.data.build_or_load_document_cache("train"),
                                    config.trainer.train_mesh_info,
                                    config.model.seq_len,
                                    microbatched=True)

    eval_dataset = ShardedIndexedDataset(config.data.build_or_load_document_cache("validation"),
                                         config.trainer.eval_mesh_info,
                                         config.model.seq_len,
                                         microbatched=False)

    with config.trainer.device_mesh as mesh:

        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        resource_partitions = {
            "hidden": ResourceAxis.MODEL,
            # "mlp": ResourceAxis.MODEL,
            "batch": ResourceAxis.DATA,
        }

        # initialize the model
        vocab = Axis("vocab", len(tokenizer))
        model = Gpt2LMHeadModel(
            vocab,
            config.model,
            key=model_key)

        model_resources = infer_resource_partitions(model, resource_partitions)

        # convert to appropriate dtype
        model = jax.tree_map(lambda array: array.astype(config.dtype), model)
        model = pjit(lambda m: m, in_axis_resources=None, out_axis_resources=model_resources)(model)

        # initialize the optimizer
        optim = config.trainer.optimizer()

        # loss function
        def compute_loss(model: Gpt2LMHeadModel,
                         input_ids,
                         key):
            pred_y = model(input_ids, key)
            token_loss = jnp.mean(
                optax.softmax_cross_entropy(pred_y[:-1], jax.nn.one_hot(input_ids[1:], num_classes=tokenizer.vocab_size)))

            return token_loss

        compute_loss_vmap = vmap(compute_loss, in_axes=[None, 0, 0])

        def mean_loss(model: Gpt2LMHeadModel, input_ids, key):
            return jnp.mean(compute_loss_vmap(model, input_ids, key))

        compute_loss_pjit = pjit(partial(mean_loss, key=None),
                                 in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
                                 out_axis_resources=None)

        # get the gradient using a wrapper around jax.value_and_grad
        compute_loss_and_grad = eqx.filter_value_and_grad(mean_loss)

        # boilerplate hooks and such
        engine = TrainerHooks()

        wandb.config['parameter_count'] = parameter_count(model)
        wandb.summary['parameter_count'] = parameter_count(model)

        flops_estimate = levanter.jax_utils.flops_estimate(mean_loss, model, jnp.zeros((1, config.model.seq_len), dtype=jnp.uint32), None)
        wandb.summary['flops_per_example'] = flops_estimate

        engine.add_hook(pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(log_to_wandb, every=1)
        engine.add_hook(log_performance_stats(config.model.seq_len,
                                              config.trainer.train_batch_size,
                                              flops_per_example=flops_estimate), every=1)

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 50):
                yield (batch, )

        evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)
        engine.add_hook(evaluate, every=config.trainer.steps_per_eval)
        # TODO: model sharded saving
        save = callbacks.save_model(checkpoint_dir)
        engine.add_hook(save, every=config.trainer.steps_per_save)

        # data loader
        iter_data = iter(dataset)

        # as with most things jax, the optimizer is a function that takes a model and an optimizer state returns a new
        # model and optimizer state
        opt_state = optim.init(model)

        # load the last checkpoint and resume if we want
        # TODO: need to seek in dataloader
        # TODO: wandb resume logic?
        resume_step = None
        if config.trainer.load_last_checkpoint:
            checkpoint = load_checkpoint(model, (opt_state, training_key),
                                         config.trainer.load_checkpoint_path or run_dir)
            if checkpoint is not None:
                model, (opt_state, training_key), resume_step = checkpoint
            elif config.trainer.load_checkpoint_path:
                raise ValueError("No checkpoint found")
            else:
                print("No checkpoint found. Starting from scratch")

        if resume_step is not None:
            # step is after the batch, so we need to seek to step
            # TODO: iter_data.seek(resume_step +1)
            for _ in range(resume_step + 1):
                next(iter_data)
            resume_step = resume_step + 1
        else:
            resume_step = 0

        # input_ids is [microsteps, batch_axis, per_device_batch, ...]
        # keys are [microsteps, batch_axis, model_axis, per_device_batch, ...]
        def train_step(model, opt_state, input_ids, keys):
            loss, grads = accumulate_gradients(compute_loss_and_grad, model, input_ids, keys)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        opt_state_axes = infer_resource_partitions(opt_state, resource_partitions)

        train_step = pjit(train_step,
                          in_axis_resources=(model_resources, opt_state_axes, PartitionSpec(None, ResourceAxis.DATA, None), None),
                          out_axis_resources=(None, model_resources, opt_state_axes))

        train_mesh_info = config.trainer.train_mesh_info

        for step in range(resume_step, config.trainer.num_train_steps):
            time_in = time.perf_counter()
            my_key, training_key = jrandom.split(training_key, 2)

            input_ids = next(iter_data)

            # take just the examples for this rank
            micro_keys = shaped_rng_split(my_key, (
                train_mesh_info.microbatches_per_step,
                train_mesh_info.data_axis_size
            ))

            step_loss, model, opt_state = train_step(model, opt_state, input_ids, micro_keys)
            step_loss = jnp.mean(step_loss).item()

            time_out = time.perf_counter()
            engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, time_out - time_in))

        last_step = StepInfo(config.trainer.num_train_steps, model, opt_state, step_loss, training_key,
                             time_out - time_in)

        try:
            evaluate(last_step)
        except:
            print("Failed to evaluate")
            import traceback
            traceback.print_exc()
            import sys
            sys.exit(1)  # leave
        save(last_step)


if __name__ == "__main__":
    main()
