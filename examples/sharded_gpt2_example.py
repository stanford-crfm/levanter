import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import datasets
import equinox as eqx
import jax
from jax.experimental.maps import xmap

from psithuros import callbacks
from psithuros.axis_names import xmapped_init, Array, infer_named_axes, LogicalAxis, ResourceAxis, \
    unwrap_axis_names
from psithuros.logging import pbar_logger, log_to_wandb
from psithuros.models.gpt2 import Gpt2Config
from psithuros.models.sharded_gpt2 import ShardedGpt2LMHeadModel

from collections import Counter

from psithuros.logging import log_performance_stats

print(Counter([type(dev) for dev in jax.devices()]))
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import optax
import pyrallis
import wandb
from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase

from psithuros.checkpoint import load_checkpoint
from psithuros.config import TrainerConfig, WandbConfig
from psithuros.data.text import IndexedDataset, batched
from psithuros.jax_utils import shaped_rng_split, parameter_count
from psithuros.modeling_utils import accumulate_gradients
from psithuros.trainer_hooks import TrainerHooks, StepInfo

# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py

@dataclass
class HfDatasetParams:
    id: str
    name: Optional[str] = None

    tokenizer: str = "gpt2"
    enforce_eos: bool = True

    def load(self, tokenizer, split, seq_len, cache_dir):
        def tokenize(texts):
            return tokenizer(texts, return_attention_mask=False)
        dataset = datasets.load_dataset(self.id, name=self.name, split=split)
        data = dataset["text"]
        if self.enforce_eos:
            data = map(lambda x: x + tokenizer.eos_token, data)
        token_iter = (tokenize(batch) for batch in batched(data, 1000))
        # TODO: sharded data loading
        return IndexedDataset.build_or_load(token_iter, f"{cache_dir}/{self.id}/{split}", seq_len)


@dataclass
class TrainGpt2Config:
    data: HfDatasetParams
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()
    cache_dir: str = "cache/"
    run_base_dir: str = "runs/"

    dtype: jnp.dtype = jnp.float32


def dataloader(dataset: IndexedDataset, tokenizer: PreTrainedTokenizerBase, batch_size, max_passes=None):
    eos = tokenizer.eos_token_id
    for i in range(max_passes or 400_000):
        for batch in batched(dataset, batch_size):
            input_ids = [jnp.array(ex["input_ids"], dtype=jnp.int32) for ex in batch]
            # TODO: add masking
            input_ids = jnp.stack(input_ids)
            outputs = jnp.concatenate([input_ids[:, 1:], jnp.full((input_ids.shape[0], 1), eos)], axis=1)

            yield input_ids, outputs


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.wandb.init(config)
    cache_dir = config.cache_dir
    run_dir = f"{config.run_base_dir}/{wandb.run.name or wandb.run.id}"

    tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    dataset = config.data.load(tokenizer, "train", config.model.seq_len, cache_dir)
    valid_dataset = config.data.load(tokenizer, "validation", config.model.seq_len, cache_dir)

    global_rank = jax.process_index()

    with config.trainer.device_mesh(data_name=ResourceAxis.DATA, model_name=ResourceAxis.MODEL) as mesh:

        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        axis_resources = {LogicalAxis.PARAMS: ResourceAxis.MODEL, LogicalAxis.BATCH: ResourceAxis.DATA}
        axis_sizes = mesh.shape
        axis_sizes = {logical: axis_sizes[physical] for logical, physical in axis_resources.items()}

        # initialize the model
        model = xmapped_init(ShardedGpt2LMHeadModel,
                             static_argnums=(0, 1),
                             axis_resources=axis_resources,
                             axis_sizes=axis_sizes)(tokenizer.vocab_size, config.model, key=jrandom.split(model_key, axis_sizes[LogicalAxis.PARAMS]))

        model_axes = infer_named_axes(model, ShardedGpt2LMHeadModel)
        # model_axes returns a pytree of AxisNames, but xmap needs the inner names
        model_axes = unwrap_axis_names(model_axes)

        # convert to appropriate dtype
        model = jax.tree_map(lambda array: array.astype(config.dtype), model)

        # initialize the optimizer
        optim = config.trainer.optimizer()

        # loss function
        def compute_loss(model: ShardedGpt2LMHeadModel,
                         input_ids: Array[LogicalAxis.BATCH, ...],
                         targets: Array[LogicalAxis.BATCH, ...],
                         key: [LogicalAxis.PARAMS, LogicalAxis.BATCH, ...]):
            pred_y = jax.vmap(model)(input_ids, key)
            token_loss = jnp.mean(
                optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(targets, num_classes=tokenizer.vocab_size)))
            return jax.lax.pmean(token_loss, LogicalAxis.BATCH)

        compute_loss_xmap = xmap(partial(compute_loss, key=None),
                                 in_axes=[
                                     model_axes,
                                     [LogicalAxis.BATCH, ...],
                                     [LogicalAxis.BATCH, ...],
                                 ],
                                 out_axes=[...],
                                 axis_resources=axis_resources)


        # get the gradient using a wrapper around jax.value_and_grad
        compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)

        def compute_and_reduce_grads(model: ShardedGpt2LMHeadModel,
                                     input_ids: Array[LogicalAxis.BATCH, ...],
                                     targets: Array[LogicalAxis.BATCH, ...],
                                     key: [LogicalAxis.PARAMS, LogicalAxis.BATCH, ...]):
            loss, grad = compute_loss_and_grad(model, input_ids, targets, key)
            grad = jax.lax.pmean(grad, LogicalAxis.BATCH)

            # TODO: this is so gross. there has to be a better way!
            def still_has_shard_axis(x):
                try:
                    return LogicalAxis.PARAMS in x.aval.named_shaped
                except AttributeError:
                    return False
            grad = jax.tree_map(lambda x: x if still_has_shard_axis(x) else jax.lax.pmean(x, LogicalAxis.PARAMS), grad)
            return loss, grad

        # boilerplate hooks and such
        engine = TrainerHooks()

        wandb.config['parameter_count'] = parameter_count(model)
        wandb.summary['parameter_count'] = parameter_count(model)

        engine.add_hook(pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(log_to_wandb, every=1)
        engine.add_hook(log_performance_stats(config.model.seq_len, config.trainer.train_batch_size))

        def eval_dataloader():
            test_loader = dataloader(valid_dataset, tokenizer, config.trainer.per_process_eval_batch_size,
                                     max_passes=1)
            for input_ids, targets in test_loader:
                eval_shape = (config.trainer.per_process_data_axis_size,
                 config.trainer.per_device_eval_batch_size) + input_ids.shape[1:]
                input_ids = input_ids.reshape(eval_shape)
                targets = targets.reshape(eval_shape)

                yield input_ids, targets

        evaluate = callbacks.compute_validation_loss(compute_loss_xmap, eval_dataloader)
        engine.add_hook(evaluate, every=config.trainer.steps_per_eval)
        # TODO: model sharded saving
        # save = callbacks.save_model(run_dir, prepare_fn=partial(psithuros.callbacks.get_nth_rank, rank=0))
        # engine.add_hook(save, every=config.trainer.steps_per_save)

        # data loader
        iter_data = dataloader(dataset, tokenizer, config.trainer.train_batch_size)

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

        # input_ids and targets are [microsteps, batch_axis, per_device_batch, ...]
        # keys are [microsteps, batch_axis, model_axis, per_device_batch, ...]
        def train_step(model, opt_state, input_ids, targets, keys):
            loss, grads = accumulate_gradients(compute_and_reduce_grads, model, input_ids, targets, keys)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        opt_state_axes = infer_named_axes(opt_state, None)
        opt_state_axes = unwrap_axis_names(opt_state_axes)

        train_step = xmap(train_step,
                          in_axes=[model_axes, opt_state_axes, [None, LogicalAxis.BATCH, ...],
                                   [None, LogicalAxis.BATCH, ...],
                                   [None, LogicalAxis.BATCH, LogicalAxis.PARAMS, ...]],
                          out_axes=((...,), model_axes, opt_state_axes),
                          axis_resources=axis_resources)

        for step in range(resume_step, config.trainer.num_train_steps):
            time_in = time.perf_counter()
            my_key, training_key = jrandom.split(training_key, 2)

            input_ids, targets = next(iter_data)

            # take just the examples for this rank
            # TODO: refactor this to its own thing
            input_ids = input_ids[global_rank*config.trainer.per_process_train_batch_size:(global_rank+1)*config.trainer.per_process_train_batch_size]
            targets = targets[global_rank*config.trainer.per_process_train_batch_size:(global_rank+1)*config.trainer.per_process_train_batch_size]

            micro_step_shape = (config.trainer.train_microbatches_per_step,
                                config.trainer.per_process_data_axis_size,
                                config.trainer.per_device_train_batch_size) + input_ids.shape[1:]
            input_ids = input_ids.reshape(micro_step_shape)
            targets = targets.reshape(micro_step_shape)

            micro_keys = shaped_rng_split(my_key, (
                config.trainer.train_microbatches_per_step,
                config.trainer.per_process_data_axis_size,
                config.trainer.per_process_model_axis_size,
                config.trainer.per_device_train_batch_size,
                ))

            step_loss, model, opt_state = train_step(model, opt_state, input_ids, targets, micro_keys)
            step_loss = jnp.mean(step_loss).item()

            time_out = time.perf_counter()
            engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, time_out - time_in))

        last_step = StepInfo(config.trainer.num_train_steps, model, opt_state, step_loss, training_key, time_out - time_in)
        evaluate(last_step)
        # save(last_step)


if __name__ == "__main__":
    # with jax.profiler.trace("logs"):
    main()
