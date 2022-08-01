import itertools
import time
from collections import Counter
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
from jax import vmap
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import xmap

from hapax import Axis
from psithuros import callbacks
from psithuros.axis_names import xmapped_init, Array, infer_named_axes, LogicalAxis, ResourceAxis, \
    unwrap_axis_names
from psithuros.data import CachedLMDatasetConfig
from psithuros.data.sharded import ShardedIndexedDataset
from psithuros.logging import log_performance_stats
from psithuros.logging import pbar_logger, log_to_wandb
from psithuros.models.named_gpt2 import Gpt2LMHeadModel, Gpt2Config

print(Counter([type(dev) for dev in jax.devices()]))
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import optax
import pyrallis
import wandb
from transformers import GPT2Tokenizer

from psithuros.checkpoint import load_checkpoint
from psithuros.config import TrainerConfig, WandbConfig
from psithuros.jax_utils import shaped_rng_split, parameter_count
from psithuros.modeling_utils import accumulate_gradients
from psithuros.trainer_hooks import TrainerHooks, StepInfo


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()
    cache_dir: str = "cache/"
    run_base_dir: str = "runs/"

    dtype: jnp.dtype = jnp.float32


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.wandb.init(config)
    run_dir = f"{config.run_base_dir}/{wandb.run.name or wandb.run.id}"

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer
    # dataset = ShardedIndexedDataset(config.data.build_or_load_document_cache("train"),
    #                                 config.trainer.train_mesh_info,
    #                                 config.model.seq_len,
    #                                 microbatched=True)
    #
    # eval_dataset = ShardedIndexedDataset(config.data.build_or_load_document_cache("validation"),
    #                                       config.trainer.eval_mesh_info,
    #                                       config.model.seq_len,
    #                                       microbatched=False)

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

        # TODO: pjit axes
        # axis_sizes = mesh.shape
        # axis_sizes = {logical: axis_sizes[physical] for logical, physical in axis_resources.items()}

        # initialize the model
        vocab = Axis("vocab", len(tokenizer))
        model = Gpt2LMHeadModel(vocab, config.model, key = model_key)

        # convert to appropriate dtype
        model = jax.tree_map(lambda array: array.astype(config.dtype), model)

        # initialize the optimizer
        optim = config.trainer.optimizer()

        # loss function
        def compute_loss(model: Gpt2LMHeadModel,
                         input_ids: jnp.ndarray,
                         key):
            pred_y = model(input_ids, key)
            token_loss = jnp.mean(
                optax.softmax_cross_entropy(pred_y[:-1], jax.nn.one_hot(input_ids[1:], num_classes=tokenizer.vocab_size)))
            return jax.lax.pmean(token_loss, axis_name="batch")

        compute_loss_vmap = vmap(compute_loss, in_axes=(None, 0, 0), axis_name="batch")

        # get the gradient using a wrapper around jax.value_and_grad
        compute_loss_and_grad = eqx.filter_vmap(eqx.filter_value_and_grad(compute_loss, reduce_axes=("batch",)), args=(None, 0, 0), axis_name="batch")

        def compute_and_reduce_grads(model: Gpt2LMHeadModel,
                                     input_ids: jnp.ndarray,
                                     key):
            loss, grad = compute_loss_and_grad(model, input_ids, key)
            loss = jnp.mean(loss)
            grad = jax.tree_map(lambda arr: jnp.mean(arr, axis=0), grad)

            return loss, grad

        # boilerplate hooks and such
        engine = TrainerHooks()

        wandb.config['parameter_count'] = parameter_count(model)
        wandb.summary['parameter_count'] = parameter_count(model)

        engine.add_hook(pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(log_to_wandb, every=1)
        engine.add_hook(log_performance_stats(config.model.seq_len, config.trainer.train_batch_size))

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 50):
                yield (batch, None)

        evaluate = callbacks.compute_validation_loss(compute_loss_vmap, eval_dataloader)
        engine.add_hook(evaluate, every=config.trainer.steps_per_eval)
        # TODO: model sharded saving
        # save = callbacks.save_model(run_dir, prepare_fn=partial(psithuros.callbacks.get_nth_rank, rank=0))
        # engine.add_hook(save, every=config.trainer.steps_per_save)

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

        # because we're running the model on each device in parallel, we need to make sure the model is on each device
        # (and the optimization state too)
        # model = jax.device_put_replicated(model, devices)
        # opt_state = jax.device_put_replicated(opt_state, devices)

        # input_ids is [microsteps, batch_axis, per_device_batch, ...]
        # keys are [microsteps, batch_axis, model_axis, per_device_batch, ...]
        def train_step(model, opt_state, input_ids, keys):
            loss, grads = accumulate_gradients(compute_and_reduce_grads, model, input_ids, keys)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        # train_step = xmap(train_step,
        #                   in_axes=[model_axes, opt_state_axes, [None, LogicalAxis.BATCH, ...],
        #                            [None, LogicalAxis.BATCH, LogicalAxis.PARAMS, ...]],
        #                   out_axes=((...,), model_axes, opt_state_axes),
        #                   axis_resources=axis_resources)

        train_mesh_info = config.trainer.train_mesh_info

        for step in range(resume_step, config.trainer.num_train_steps):
            time_in = time.perf_counter()
            my_key, training_key = jrandom.split(training_key, 2)

            input_ids: GlobalDeviceArray = next(iter_data)

            # take just the examples for this rank
            micro_keys = shaped_rng_split(my_key, (
                train_mesh_info.microbatches_per_step,
                train_mesh_info.local_data_axis_size,
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
        # save(last_step)


if __name__ == "__main__":
    # with jax.profiler.trace("logs"):
    main()
