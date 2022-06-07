from dataclasses import dataclass
from functools import partial
from typing import Optional


import datasets
import equinox as eqx
import jax
print(jax.devices())
import jax.lax as lax
import jax.numpy as jnp
import jax.profiler
import jax.random as jrandom
import optax
import pyrallis
import wandb
from jax import pmap
from tqdm import tqdm
from transformers import GPT2Config, AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerBase

from psithuros.checkpoint import load_checkpoint, save_checkpoint
from psithuros.config import TrainerConfig, WandbConfig
from psithuros.data.text import IndexedDataset, batched
from psithuros.jax_utils import shaped_rng_split, replicate, fold_left
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
    for i in range(max_passes or 400_000):
        for batch in batched(dataset, batch_size):
            input_ids = [jnp.array(ex["input_ids"], dtype=jnp.int32) for ex in batch]
            input_ids = jnp.stack(input_ids)
            outputs = jnp.concatenate([input_ids[:, 1:], jnp.full((input_ids.shape[0], 1), eos)], axis=1)

            yield input_ids, outputs


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

    model = jax.tree_map(lambda x: x.astype(config.dtype), model)

    optim = config.trainer.optimizer()
    opt_state = optim.init(model)

    @jax.profiler.annotate_function
    def compute_loss(model, input_ids, targets, key, inference):
        model = partial(model, inference=inference, key=key)
        pred_y = jax.vmap(model)(input_ids)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(targets, num_classes=tokenizer.vocab_size)))

    compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)
    compute_loss_and_grad = pmap(compute_loss_and_grad, axis_name="device", in_axes=(None, 0, 0, 0, None), static_broadcasted_argnums=(4))
    compute_loss_eval = pmap(compute_loss, "device", in_axes=(None, 0, 0, 0, None), static_broadcasted_argnums=(4))

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
        # have to dereplicate the model and opt states
        # TODO: when we do model sharding we have to do something cleverer
        model = info.model
        # model = jax.tree_map(lambda x: x[0], model)
        model = jax.device_get(model)

        opt_state = info.opt_state
        # opt_state = jax.tree_map(lambda x: x[0], opt_state)
        opt_state = jax.device_get(opt_state)

        save_checkpoint(model, (opt_state, info.next_key), step, f"{run_dir}/step-{info.step}")

    iter_data = dataloader(dataset, tokenizer, config.trainer.train_batch_size)

    # load the last checkpoint
    # TODO: add support for sharding
    # TODO write some tests for serialization
    # TODO: need to seek in dataloader
    # TODO: wandb resume logic?
    resume_step = None
    if config.trainer.load_last_checkpoint:
        checkpoint = load_checkpoint(model, (opt_state, training_key), config.trainer.load_checkpoint_path or run_dir)
        if checkpoint is not None:
            model, (opt_state, training_key), resume_step = checkpoint
        elif config.trainer.load_checkpoint_path:
            raise ValueError("No checkpoint found")
        else:
            print("No checkpoint found. Starting from scratch")

    if resume_step is not None:
        # step is after the batch, so we need to seek to step
        resume_step_with_micro = (resume_step + 1) * config.trainer.train_microbatches_per_step
        # TODO: iter_data.seek(resume_step +1)
        for _ in range(resume_step_with_micro):
            next(iter_data)
    else:
        resume_step = 0

    @eqx.filter_jit
    def train_step(model, opt_state, input_ids, targets, keys):
        loss_total = jnp.zeros(())
        grad_totals = jax.tree_map(jnp.zeros_like, model)

        def accumulate(acc, x):
            loss, grads = compute_loss_and_grad(model, *x, False)
            loss_acc, grad_acc = acc
            loss_acc += jnp.mean(loss)
            grad_acc = jax.tree_map(lambda totals, new_grads: totals + jnp.mean(new_grads, axis=0), grad_acc, grads)

            return loss_acc, grad_acc

        loss_total, grad_totals = fold_left(accumulate, (loss_total, grad_totals), (input_ids, targets, keys))

        loss_total = loss_total/config.trainer.train_microbatches_per_step
        grad_totals = jax.tree_map(lambda x: x / config.trainer.train_microbatches_per_step, grad_totals)

        updates, opt_state = optim.update(grad_totals, opt_state)
        model = eqx.apply_updates(model, updates)

        return loss_total, model, opt_state

    loss = RunningMean()
    for step in range(resume_step, config.trainer.num_train_steps):
        input_ids, targets = next(iter_data)
        micro_step_shape = (config.trainer.train_microbatches_per_step, len(devices), config.trainer.per_device_train_batch_size) + input_ids.shape[1:]
        input_ids = input_ids.reshape(micro_step_shape)
        targets = targets.reshape(micro_step_shape)
        my_key, training_key = jrandom.split(training_key, 2)
        micro_keys = shaped_rng_split(my_key, (config.trainer.train_microbatches_per_step, len(devices)))
        step_loss, model, opt_state = train_step(model, opt_state, input_ids, targets, micro_keys)
        loss.update(step_loss)
        engine.run_hooks(StepInfo(step, model, opt_state, loss.mean.item(), training_key))

    evaluate(StepInfo(config.trainer.num_train_steps, model, opt_state, loss.mean, training_key))


if __name__ == "__main__":
    # with jax.profiler.trace("logs"):
    main()
