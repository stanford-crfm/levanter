import os
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import pyrallis
import wandb
from jax import pmap
from tqdm import tqdm
from transformers import GPT2Config

import psithuros
from psithuros.config import TrainerConfig, WandbConfig
from psithuros.logging import log_optimizer_hyperparams
from psithuros.modeling_utils import RunningMean
from psithuros.jax_utils import shaped_rng_split, replicate
from psithuros.models.gpt2 import Gpt2LMHeadModel

# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py

def dataloader(arrays, batch_size, *, key, max_passes=None):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    i = 0
    while max_passes is None or i < max_passes:
        i += 1
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, seq_len, vocab_size, *, key):
    k_x, k_y = jrandom.split(key, 2)
    x = jrandom.randint(k_x, [dataset_size, seq_len], minval=0, maxval=vocab_size)
    y = jnp.concatenate( [x[:, 1:], jnp.zeros((dataset_size, 1), dtype=jnp.int32)], axis=1)

    return x, y

@dataclass
class MyConfig:
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()

    # data params
    dataset_size: int = 10000
    seq_len: int = 512
    vocab_size: int = 2048


@pyrallis.wrap()
def main(config: MyConfig):
    config.wandb.init(config)

    seed = config.trainer.seed

    data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    xs, ys = get_data(config.dataset_size, config.seq_len, config.vocab_size, key=data_key)

    gpt_config = GPT2Config(vocab_size=config.vocab_size, n_positions=config.seq_len, n_embd=128, n_ctx=config.seq_len, n_layer=4, n_head=4, n_embd_shared_axes=0, hidden_dim=128, num_attention_heads=4, intermediate_size=128, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=config.vocab_size, type_vocab_size=2, initializer_range=0.02)

    model = Gpt2LMHeadModel(gpt_config, key=model_key)

    def compute_loss(model, x, y, key, inference):
        model = partial(model, inference=inference, key=key)
        pred_y = jax.vmap(model)(x)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(y, num_classes=config.vocab_size)))

    compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)

    def take_train_step(model, x, y, opt_state, key):
        loss, grads = compute_loss_and_grad(model, x, y, key, inference=False)
        loss = lax.pmean(loss, "device")
        grads = lax.pmean(grads, "device")
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    train_step = pmap(take_train_step, "device", in_axes=0)

    devices = config.trainer.devices()

    optim = config.trainer.optimizer()
    opt_state = optim.init(model)

    iter_data = dataloader((xs, ys),  len(devices) * config.trainer.per_device_train_batch_size, key=loader_key)

    model = replicate(model, devices)
    opt_state = replicate(opt_state, devices)
    pbar = tqdm(range(config.trainer.num_train_steps), desc="train", total=config.trainer.num_train_steps)
    for step in pbar:
        loss = RunningMean(shape=1)

        # TODO: factor out optimizer logging
        log_optimizer_hyperparams(opt_state, step=step)
        # wandb.log({"learning_rate": opt_state.hyperparams['learning_rate']}, step=step)

        for micro_step in range(config.trainer.train_microbatches_per_step):
            # TODO: replicate data loader instead?
            x, y = next(iter_data)

            my_key, training_key = jrandom.split(training_key, 2)
            micro_keys = shaped_rng_split(my_key, (len(devices),))
            micro_step_shape = (len(devices), config.trainer.per_device_train_batch_size) + x.shape[1:]

            x = x.reshape(micro_step_shape)
            y = y.reshape(micro_step_shape)

            my_loss, model, opt_state = train_step(model, x, y, opt_state, micro_keys)

            loss.update(jnp.mean(my_loss))

        loss = loss.value.item()
        wandb.log({"train/loss": loss}, step=step)
        pbar.set_postfix({"loss": loss})

    del pbar

    total_loss = RunningMean(shape=1)
    test_loader = dataloader((xs, ys), config.trainer.per_device_eval_batch_size * len(devices), max_passes=1, key=loader_key)

    compute_loss = pmap(compute_loss, "device", in_axes=0, static_broadcasted_argnums=(4))
    pbar = tqdm(test_loader, desc="eval", total=len(xs) // (config.trainer.per_device_eval_batch_size * len(devices)))
    for (x, y) in pbar:
        my_key, training_key = jrandom.split(training_key, 2)

        micro_step_shape = (len(devices), config.trainer.per_device_train_batch_size) + x.shape[1:]
        micro_keys = shaped_rng_split(my_key, (len(devices),))
        x = x.reshape(micro_step_shape)
        y = y.reshape(micro_step_shape)

        loss = compute_loss(model, x, y, micro_keys, True)
        loss = jnp.mean(loss).item()
        total_loss.update(loss)

    total_loss = total_loss.value.item()
    wandb.log({"test/loss": total_loss})

    print(f"Final total loss {total_loss}")


if __name__ == "__main__":
    main()
