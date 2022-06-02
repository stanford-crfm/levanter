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
from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase

from psithuros.config import TrainerConfig, WandbConfig
from psithuros.data.text import IndexedDataset, batched
from psithuros.jax_utils import shaped_rng_split, replicate
from psithuros.logging import log_optimizer_hyperparams
from psithuros.modeling_utils import RunningMean
from psithuros.models.palm_lite import PaLM


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
class TrainPalmConfig:
    data: DataParams
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    cache_dir: str = "cache/"

    seq_len: int = 512


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
def main(config: TrainPalmConfig):
    config.wandb.init(config)
    cache_dir = config.cache_dir

    seed = config.trainer.seed

    data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    dataset = config.data.load(tokenizer, "train", config.seq_len, cache_dir)
    valid_dataset = config.data.load(tokenizer, "validation", config.seq_len, cache_dir)

    model = PaLM(
        num_tokens=len(tokenizer),
        # dim=512,
        # dim_head=256,
        # depth=12,
        # heads=12,
        dim=512,
        dim_head=256,
        depth=12,
        heads=12,
        key=model_key,
        max_seq_len=config.seq_len,
    )


    @jax.profiler.annotate_function
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(y, num_classes=tokenizer.vocab_size)))

    compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)

    @jax.profiler.annotate_function
    def take_train_step(model, x, y, opt_state):
        loss, grads = compute_loss_and_grad(model, x, y)
        loss = lax.pmean(loss, "device")
        grads = lax.pmean(grads, "device")
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    train_step = pmap(take_train_step, "device", in_axes=0)

    devices = config.trainer.devices()

    optim = config.trainer.optimizer()
    opt_state = optim.init(model)

    iter_data = dataloader(dataset, tokenizer, len(devices) * config.trainer.per_device_train_batch_size)

    model = replicate(model, devices)
    opt_state = replicate(opt_state, devices)
    pbar = tqdm(range(config.trainer.num_train_steps), desc="train", total=config.trainer.num_train_steps)
    for step in pbar:
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            loss = RunningMean(shape=1)

            log_optimizer_hyperparams(opt_state, step=step)

            for micro_step in range(config.trainer.train_microbatches_per_step):
                # TODO: replicate data loader instead?
                with jax.profiler.TraceAnnotation("data loading"):
                    x, y = next(iter_data)

                    my_key, training_key = jrandom.split(training_key, 2)
                    micro_keys = shaped_rng_split(my_key, (len(devices),))
                    micro_step_shape = (len(devices), config.trainer.per_device_train_batch_size) + x.shape[1:]

                    x = x.reshape(micro_step_shape).block_until_ready()
                    y = y.reshape(micro_step_shape).block_until_ready()

                # with jax.profiler.TraceAnnotation("actual step"):
                my_loss, model, opt_state = train_step(model, x, y, opt_state)# print([b.device() for b in model.lm_head.device_buffers])
                model#.lm_head.block_until_ready()

                loss.update(jnp.mean(my_loss))

            loss = loss.value.item()
            wandb.log({"train/loss": loss}, step=step)
            pbar.set_postfix({"loss": loss})

    del pbar

    total_loss = RunningMean(shape=1)
    test_loader = dataloader(valid_dataset, tokenizer, config.trainer.per_device_eval_batch_size * len(devices), max_passes=1)

    compute_loss = pmap(compute_loss, "device", in_axes=0)
    pbar = tqdm(test_loader, desc="eval")
    for (x, y) in pbar:
        my_key, training_key = jrandom.split(training_key, 2)

        micro_step_shape = (len(devices), config.trainer.per_device_train_batch_size) + x.shape[1:]
        micro_keys = shaped_rng_split(my_key, (len(devices),))
        x = x.reshape(micro_step_shape)
        y = y.reshape(micro_step_shape)

        loss = compute_loss(model, x, y)
        loss = jnp.mean(loss).item()
        total_loss.update(loss)

    total_loss = total_loss.value.item()
    wandb.log({"eval/loss": total_loss})

    print(f"Final total loss {total_loss}")


if __name__ == "__main__":
    # with jax.profiler.trace("logs"):
    main()
