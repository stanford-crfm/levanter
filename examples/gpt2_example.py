import glob
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional

import datasets
import equinox as eqx
import jax
import jax.profiler
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import pyrallis
print(jax.devices())
import wandb
from jax import pmap
from tqdm import tqdm
from transformers import GPT2Config, AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase

from psithuros.config import TrainerConfig, WandbConfig
from psithuros.data.text import IndexedDataset, batched
from psithuros.jax_utils import shaped_rng_split, replicate
from psithuros.logging import log_optimizer_hyperparams
from psithuros.modeling_utils import RunningMean
from psithuros.models.gpt2 import Gpt2LMHeadModel
from psithuros.trainer_hooks import TrainerHooks, StepInfo  # , engine_from_loss_fn


# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py

@dataclass
class DataParams:
    id: str
    name: Optional[str] = None

    tokenizer: str = "gpt2"

    def load(self, tokenizer, split, seq_len, cache_dir):
        dataset = datasets.load_dataset(self.id, name=self.name, split=split)
        token_iter = (tokenizer(batch) for batch in batched(dataset["text"], 1000))
        return IndexedDataset.build_or_load(token_iter, f"{cache_dir}/{self.id}/{split}", seq_len)


@dataclass
class TrainGpt2Config:
    data: DataParams
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    cache_dir: str = "cache/"
    run_base_dir: str = "runs/"

    seq_len: int = 512
    dtype: jnp.dtype = jnp.float32


def dataloader(dataset: IndexedDataset, tokenizer: PreTrainedTokenizerBase, batch_size, max_passes=None):
    eos = tokenizer.eos_token_id
    # batch = next(batched(dataset, batch_size))
    for i in range(max_passes or 400_000):
        for batch in batched(dataset, batch_size):
            input_ids = [jnp.array(ex["input_ids"], dtype=jnp.int32) for ex in batch]
            input_ids = jnp.stack(input_ids)
            outputs = jnp.concatenate([input_ids[:, 1:], jnp.full((input_ids.shape[0], 1), eos)], axis=1)

            yield input_ids, outputs
            # yield input_ids, input_ids


@pyrallis.wrap()
def main(config: TrainGpt2Config):
    config.wandb.init(config)
    cache_dir = config.cache_dir
    run_dir = f"{config.run_base_dir}/{wandb.run.name or wandb.run.id}"

    seed = config.trainer.seed

    data_key, loader_key, model_key, training_key, eval_key = jrandom.split(jrandom.PRNGKey(seed), 5)
    tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    dataset = config.data.load(tokenizer, "train", config.seq_len, cache_dir)
    valid_dataset = config.data.load(tokenizer, "validation", config.seq_len, cache_dir)

    gpt_config = GPT2Config(vocab_size=tokenizer.vocab_size,
                            n_positions=config.seq_len,
                            n_ctx=config.seq_len,
                            n_embd=128,
                            n_layer=4,
                            n_head=4
                            )

    model = Gpt2LMHeadModel(gpt_config, key=model_key)

    model = jax.tree_map(lambda input_ids: input_ids.astype(config.dtype), model)

    optim = config.trainer.optimizer()
    opt_state = optim.init(model)

    @jax.profiler.annotate_function
    def compute_loss(model, input_ids, targets, key, inference):
        model = partial(model, inference=inference, key=key)
        pred_y = jax.vmap(model)(input_ids)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(targets, num_classes=tokenizer.vocab_size)))

    compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)
    compute_loss_eval = pmap(compute_loss, "device", in_axes=0, static_broadcasted_argnums=(4))

    @jax.profiler.annotate_function
    def take_train_step(model, input_ids, targets, opt_state, key):
        loss, grads = compute_loss_and_grad(model, input_ids, targets, key, inference=False)
        loss = lax.pmean(loss, "device")
        grads = lax.pmean(grads, "device")
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    train_step = pmap(take_train_step, "device", in_axes=0)

    devices = config.trainer.devices()

    engine = TrainerHooks()

    pbar = tqdm(range(config.trainer.num_train_steps), desc="train", total=config.trainer.num_train_steps)

    @engine.add_hook(every=1)
    def update_pbar(step: StepInfo):
        pbar.update(1)
        pbar.set_postfix(loss=step.loss)

    @engine.add_hook(every=1)
    def log_to_wandb(step: StepInfo):
        wandb.log({"train/loss": step.loss}, step=step.step)

    @engine.add_hook(every=config.trainer.steps_per_eval)
    def evaluate(info: StepInfo):
        nonlocal eval_key
        total_loss = RunningMean(shape=1)
        test_loader = dataloader(valid_dataset, tokenizer, config.trainer.per_device_eval_batch_size * len(devices),
                                 max_passes=1)

        pbar = tqdm(test_loader, desc="eval", position=1, leave=False)
        for (input_ids, targets) in pbar:
            my_key, eval_key = jrandom.split(eval_key, 2)

            micro_step_shape = (len(devices), config.trainer.per_device_train_batch_size) + input_ids.shape[1:]
            micro_keys = shaped_rng_split(my_key, (len(devices),))
            input_ids = input_ids.reshape(micro_step_shape)
            targets = targets.reshape(micro_step_shape)

            loss = compute_loss_eval(info.model, input_ids, targets, micro_keys, True)
            loss = jnp.mean(loss).item()
            total_loss.update(loss)
            pbar.set_postfix(loss=total_loss.mean.item())

        total_loss = total_loss.mean.item()
        wandb.log({"eval/loss": total_loss}, step=info.step)

    @engine.add_hook(every=config.trainer.steps_per_save)
    def save(info: StepInfo):
        os.makedirs(f"{run_dir}/step-{info.step}", exist_ok=True)
        # have to dereplicate the model and opt states
        # TODO: when we do model sharding we have to do something cleverer
        model = info.model
        model = jax.tree_map(lambda input_ids: input_ids[0], model)
        model = jax.device_get(model)

        model_path = f"{run_dir}/step-{info.step}/model.eqx"
        eqx.tree_serialise_leaves(model_path, model)

        opt_state = info.opt_state
        opt_state = jax.tree_map(lambda input_ids: input_ids[0], opt_state)
        opt_state = jax.device_get(opt_state)

        opt_path = f"{run_dir}/step-{info.step}/opt_state.eqx"
        eqx.tree_serialise_leaves(opt_path, opt_state)

        # have to save the key too. it's just a numpy array?
        key_path = f"{run_dir}/step-{info.step}/key.npy"
        jax.numpy.save(key_path, info.next_key)

    # load function to go with it
    def load(ckpt_dir, model, opt_state):
        model = eqx.tree_deserialise_leaves(f"{ckpt_dir}/model.eqx", model)
        opt_state = eqx.tree_deserialise_leaves(f"{ckpt_dir}/opt_state.eqx", opt_state)
        key = jax.numpy.load(f"{ckpt_dir}/key.npy")
        return model, opt_state, key

    # load the last checkpoint
    if config.trainer.load_last_checkpoint:
        # first check if it's an actual checkpoint dir, or a dir of checkpoint dirs
        if config.trainer.load_checkpoint_path and os.path.exists(f"{config.trainer.load_checkpoint_path}/model.eqx"):
            checkpoint_path = config.trainer.load_checkpoint_path
        else:
            checkpoints_dir = config.trainer.load_checkpoint_path or run_dir
            # load the last checkpoint
            ckpt_dirs = sorted(glob.glob(f"{checkpoints_dir}/*"))
            if len(ckpt_dirs) > 0:
                checkpoint_path = ckpt_dirs[-1]
            else:
                checkpoint_path = None

        if checkpoint_path:
            model, opt_state, key = load(checkpoint_path, model, opt_state)
            print(f"loaded checkpoint from {checkpoint_path}")
        elif config.trainer.load_checkpoint_path:
            raise FileNotFoundError(f"Could not find checkpoint at {config.trainer.load_checkpoint_path}")

    # replicate to all devices to make pmap happy
    model = replicate(model, devices)
    opt_state = replicate(opt_state, devices)

    # add support for sharding
    # TODO write some tests for serialization
    # TODO: need to seek in dataloader
    # TODO: wandb resume logic

    loss = RunningMean(shape=1)
    iter_data = dataloader(dataset, tokenizer, len(devices) * config.trainer.per_device_train_batch_size)
    for step in range(config.trainer.num_train_steps):
        for micro_step in range(config.trainer.train_microbatches_per_step):
            # TODO: replicate data loader instead?
            with jax.profiler.TraceAnnotation("data loading"):
                input_ids, targets = next(iter_data)

                my_key, training_key = jrandom.split(training_key, 2)
                micro_keys = shaped_rng_split(my_key, (len(devices),))
                micro_step_shape = (len(devices), config.trainer.per_device_train_batch_size) + input_ids.shape[1:]

                input_ids = input_ids.reshape(micro_step_shape)
                targets = targets.reshape(micro_step_shape)

            my_loss, model, opt_state = train_step(model, input_ids, targets, opt_state, micro_keys)

            loss.update(jnp.mean(my_loss))

        engine.run_hooks(StepInfo(step, model, opt_state, loss.mean, training_key))

    evaluate(StepInfo(config.trainer.num_train_steps, model, opt_state, loss.mean, training_key))


if __name__ == "__main__":
    # with jax.profiler.trace("logs"):
    main()
