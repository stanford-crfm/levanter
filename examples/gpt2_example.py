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

from haliax import Axis
from haliax.partitioning import resource_mapping
from levanter import callbacks
from levanter.axis_names import ResourceAxis, infer_resource_partitions, named_pjit
from levanter.data import CachedLMDatasetConfig
from levanter.data.sharded import ShardedIndexedDataset
from levanter.logging import log_performance_stats, log_to_wandb, pbar_logger
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


print(Counter([type(dev) for dev in jax.devices()]))
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import pyrallis
from transformers import GPT2Tokenizer

import wandb
from levanter.checkpoint import load_checkpoint
from levanter.config import TrainerConfig, WandbConfig
from levanter.jax_utils import global_key_array, parameter_count
from levanter.modeling_utils import accumulate_gradients
from levanter.trainer_hooks import StepInfo, TrainerHooks


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py


@dataclass
class TrainGpt2Config:
    data: CachedLMDatasetConfig = CachedLMDatasetConfig()
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: Gpt2Config = Gpt2Config()

    run_base_dir: str = "runs/"
    checkpoint_dir: str = "checkpoints/"

    log_z_regularization: float = 0.0


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.trainer.initialize_jax_config()

    config.wandb.init(config)
    run_name = wandb.run.name or wandb.run.id
    run_dir = f"{config.run_base_dir}/{run_name}"
    checkpoint_dir = f"{config.checkpoint_dir}/{run_name}"

    tokenizer: GPT2Tokenizer = config.data.the_tokenizer
    dataset = ShardedIndexedDataset(
        config.data.build_or_load_document_cache("train"),
        config.trainer.train_mesh_info,
        config.model.seq_len,
        microbatched=True,
    )

    eval_dataset = ShardedIndexedDataset(
        config.data.build_or_load_document_cache("validation"),
        config.trainer.eval_mesh_info,
        config.model.seq_len,
        microbatched=False,
    )

    resource_partitions = {
        "batch": ResourceAxis.DATA,
        # ZERO-3
        # "embed": ResourceAxis.DATA,
        "vocab": ResourceAxis.MODEL,
        "mlp": ResourceAxis.MODEL,
        # "qkv": ResourceAxis.MODEL,
        "heads": ResourceAxis.MODEL,
        # "total_head_dim": ResourceAxis.MODEL,
    }

    with config.trainer.device_mesh as mesh, resource_mapping(resource_partitions):

        # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
        # this makes deterministic training pretty easy
        seed = config.trainer.seed
        mp = config.trainer.mp
        data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)

        # initialize the model, and convert to appropriate dtype

        # TODO: factor this out
        vocab_size = len(tokenizer)
        # round up so that we can shard it if it's sharded
        vocab_resource_axis = resource_partitions.get("vocab")
        if vocab_resource_axis:
            vocab_axis_size = mesh.shape[vocab_resource_axis]
            vocab_size = (vocab_size + vocab_axis_size - 1) // vocab_axis_size * vocab_axis_size
        vocab = Axis("vocab", vocab_size)

        optim = config.trainer.optimizer()

        def init_state():
            model = mp.cast_to_param(Gpt2LMHeadModel(vocab, config.model, key=model_key, mp=mp))
            opt_state = optim.init(model)
            return model, opt_state

        model, opt_state = named_pjit(init_state, resource_partitions)()
        opt_state_resources = infer_resource_partitions(opt_state, resource_partitions)
        model_resources = infer_resource_partitions(model, resource_partitions)

        # loss function
        def compute_loss(model: Gpt2LMHeadModel, input_ids, key):
            inference = key is None
            pred_y = model(input_ids, inference=inference, key=key)
            pred_y = mp.cast_to_output(pred_y)

            pred_y = pred_y[:-1]
            target_y = input_ids[1:]

            logits_max = jnp.max(pred_y, axis=-1, keepdims=True)
            pred_y -= jax.lax.stop_gradient(logits_max)
            label_logits = jnp.take_along_axis(pred_y, target_y[..., None], axis=-1)[..., 0]
            log_normalizers = jnp.log(jnp.sum(jnp.exp(pred_y), axis=-1))

            loss = log_normalizers - label_logits
            loss = jnp.mean(loss)

            if not inference and config.log_z_regularization > 0:
                mean_logz = jnp.mean(log_normalizers)
                loss += config.log_z_regularization * mean_logz * mean_logz

            return loss

        compute_loss_vmap = vmap(compute_loss, in_axes=[None, 0, 0], spmd_axis_name=ResourceAxis.DATA)

        def mean_loss(model: Gpt2LMHeadModel, input_ids, key):
            return jnp.mean(compute_loss_vmap(model, input_ids, key))

        compute_loss_pjit = pjit(
            partial(mean_loss, key=None),
            in_axis_resources=(model_resources, PartitionSpec(ResourceAxis.DATA, None)),
            out_axis_resources=None,
        )

        # get the gradient using a wrapper around jax.value_and_grad
        compute_loss_and_grad = eqx.filter_value_and_grad(mean_loss)

        # boilerplate hooks and such
        engine = TrainerHooks()

        wandb.summary["parameter_count"] = parameter_count(model)

        # flops = flops_estimate(
        #    compute_loss_and_grad,
        #    model,
        #    jnp.zeros((1, config.model.seq_len), dtype=jnp.uint32),
        #    None,
        # )
        # wandb.summary["flops_per_example"] = flops

        engine.add_hook(pbar_logger(total=config.trainer.num_train_steps), every=1)
        engine.add_hook(log_to_wandb, every=1)
        engine.add_hook(
            log_performance_stats(
                config.model.seq_len,
                config.trainer.train_batch_size,
                # flops_per_example=flops,
            ),
            every=1,
        )

        def eval_dataloader():
            # TODO: only do one pass
            for batch in itertools.islice(eval_dataset, 50):
                yield (batch,)

        evaluate = callbacks.compute_validation_loss(compute_loss_pjit, eval_dataloader)
        engine.add_hook(evaluate, every=config.trainer.steps_per_eval)
        # TODO: model sharded saving
        save = callbacks.save_model(checkpoint_dir)
        engine.add_hook(save, every=config.trainer.steps_per_save)

        # data loader
        iter_data = iter(dataset)

        # load the last checkpoint and resume if we want
        # TODO: need to seek in dataloader
        # TODO: wandb resume logic?
        resume_step = None
        if config.trainer.load_last_checkpoint:
            checkpoint = load_checkpoint(
                model,
                (opt_state, training_key),
                config.trainer.load_checkpoint_path or run_dir,
            )
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

        # input_ids and keys are [microsteps, microbatch_size, ...]
        def train_step(model, opt_state, input_ids, keys):
            loss, grads = accumulate_gradients(compute_loss_and_grad, model, input_ids, keys)

            with jax.named_scope("optimizer"):
                updates, opt_state = optim.update(grads, opt_state, params=model)
                model = eqx.apply_updates(model, updates)

            return loss, model, opt_state

        # keys are sharded in the same way as input_ids
        # TODO: maybe put keys in the data iterator?
        data_resources = dataset.partition_spec

        train_step = pjit(
            train_step,
            in_axis_resources=(
                model_resources,
                opt_state_resources,
                data_resources,
                data_resources,
            ),
            out_axis_resources=(None, model_resources, opt_state_resources),
            donate_argnums=(0, 1),
        )

        train_mesh_info = config.trainer.train_mesh_info

        for step in range(resume_step, config.trainer.num_train_steps):
            time_in = time.perf_counter()
            my_key, training_key = jrandom.split(training_key, 2)

            input_ids = next(iter_data)

            # split keys into microsteps, and one for each example *on this node*
            micro_keys = global_key_array(
                my_key, dataset.batch_shape[:-1], train_mesh_info.mesh, dataset.partition_spec[:-1]
            )

            step_loss, model, opt_state = train_step(model, opt_state, input_ids, micro_keys)
            step_loss = jnp.mean(step_loss).item()

            time_out = time.perf_counter()
            engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, time_out - time_in))

        last_step = StepInfo(
            config.trainer.num_train_steps,
            model,
            opt_state,
            step_loss,
            training_key,
            time_out - time_in,
        )

        evaluate(last_step)
        save(last_step)


if __name__ == "__main__":
    main()
