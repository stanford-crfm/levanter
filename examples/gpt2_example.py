import time
from dataclasses import dataclass
from functools import partial
from typing import Optional


import datasets
import equinox as eqx
import jax

from psithuros.logging import log_optimizer_hyperparams, log_performance_stats

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
from psithuros.jax_utils import shaped_rng_split, flop_estimate
from psithuros.modeling_utils import RunningMean, accumulate_gradients, parameter_count
from psithuros.models.gpt2 import Gpt2LMHeadModel
from psithuros.trainer_hooks import TrainerHooks, StepInfo  # , engine_from_loss_fn

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
        return IndexedDataset.build_or_load(token_iter, f"{cache_dir}/{self.id}/{split}", seq_len)


@dataclass
class TrainGpt2Config:
    data: HfDatasetParams
    wandb: WandbConfig = WandbConfig()
    trainer: TrainerConfig = TrainerConfig()
    cache_dir: str = "cache/"
    run_base_dir: str = "runs/"

    seq_len: int = 512
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
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

    tokenizer: GPT2Tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    dataset = config.data.load(tokenizer, "train", config.seq_len, cache_dir)
    valid_dataset = config.data.load(tokenizer, "validation", config.seq_len, cache_dir)

    devices = config.trainer.devices()

    # randomness in jax is tightly controlled by "keys" which are the states of the random number generators
    # this makes deterministic training pretty easy
    seed = config.trainer.seed
    data_key, loader_key, model_key, training_key, eval_key = jrandom.split(jrandom.PRNGKey(seed), 5)

    # initialize the model
    gpt_config = GPT2Config(vocab_size=tokenizer.vocab_size,
                            n_positions=config.seq_len,
                            n_ctx=config.seq_len,
                            n_embd=config.hidden_dim,
                            n_layer=config.num_layers,
                            n_head=config.num_heads,
                            )

    model = Gpt2LMHeadModel(gpt_config, key=model_key)

    # convert to appropriate dtype
    model = jax.tree_map(lambda x: x.astype(config.dtype), model)

    # initialize the optimizer
    optim = config.trainer.optimizer()
    # as with most things jax, the optimizer is a function that takes a model and an optimizer state returns a new model
    # and optimizer state
    opt_state = optim.init(model)

    # loss function
    def compute_loss(model, input_ids, targets, key):
        # vmap automagically vectorizes the model over a batch dimension
        # TODO: this shouldn't be necessary, but for some reason it blows up the if o don't
        # model = partial(model, key=key)
        pred_y = jax.vmap(model, in_axes=(0, None))(input_ids, key)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(targets, num_classes=tokenizer.vocab_size)))

    # get the gradient using a wrapper around jax.value_and_grad
    compute_loss_and_grad = eqx.filter_value_and_grad(partial(compute_loss))

    # pmap is like vmap but instead just vectorizing, it also parallelizes the computation over devices
    # typically you want to pmap a vmap (and reshape)
    compute_loss_pmap = pmap(partial(compute_loss, key=None), "device")

    # boilerplate hooks and such
    engine = TrainerHooks()

    # get an estimate of flops for one example
    flops_per_example = flop_estimate(compute_loss_and_grad,
                                      model,
                                      jnp.ones((1, config.seq_len), dtype=jnp.int32),
                                      jnp.ones((1, config.seq_len), dtype=jnp.int32),
                                      model_key)
                                      # shaped_rng_split(model_key, (1,)))

    wandb.config['flops_per_example'] = flops_per_example
    wandb.config['parameter_count'] = parameter_count(model)
    wandb.summary['flops_per_example'] = flops_per_example
    wandb.summary['parameter_count'] = parameter_count(model)

    pbar = tqdm(range(config.trainer.num_train_steps), desc="train", total=config.trainer.num_train_steps)

    @engine.add_hook(every=1)
    def update_pbar(step: StepInfo):
        pbar.update(1)
        pbar.set_postfix(loss=step.loss)

    @engine.add_hook(every=1)
    def log_to_wandb(step: StepInfo):
        wandb.log({"train/loss": step.loss}, step=step.step)
        log_optimizer_hyperparams(step.opt_state, step=step.step)

    engine.add_hook(log_performance_stats(flops_per_example, config.seq_len, config.trainer.train_batch_size))

    @engine.add_hook(every=config.trainer.steps_per_eval)
    def evaluate(info: StepInfo):
        nonlocal eval_key
        total_loss = RunningMean(shape=1)
        test_loader = dataloader(valid_dataset, tokenizer, config.trainer.per_device_eval_batch_size * len(devices),
                                 max_passes=1)

        pbar = tqdm(test_loader, desc="eval", position=1, leave=False)
        for (input_ids, targets) in pbar:
            # the function is pmap(vmap(loss(model, input_ids, targets))), so we need to ensure
            # the inputs have shape (num_devices, batch_size_per_device, seq_len)
            micro_step_shape = (len(devices), config.trainer.per_device_eval_batch_size) + input_ids.shape[1:]
            input_ids = input_ids.reshape(micro_step_shape)
            targets = targets.reshape(micro_step_shape)

            loss = compute_loss_pmap(info.model, input_ids, targets)
            # this mean is over the devices, somewhat confusingly
            loss = jnp.mean(loss)
            total_loss.update(loss)
            pbar.set_postfix(loss=total_loss.mean.item())

        total_loss = total_loss.mean.item()
        wandb.log({"eval/loss": total_loss}, step=info.step)

    @engine.add_hook(every=config.trainer.steps_per_save)
    def save(info: StepInfo):
        def get_one_copy(tree): jax.device_get(jax.tree_map(lambda x: x[0], tree))
        # TODO: when we do model sharding we have to do something cleverer
        # it's actually pretty easy to save the model and the optimizer state
        # and enable resuming
        save_checkpoint(model=get_one_copy(info.model),
                        training_state=((get_one_copy(info.opt_state)), info.next_key),
                        step=step,
                        checkpoint_path=f"{run_dir}/step-{info.step}")

    # data loader
    iter_data = dataloader(dataset, tokenizer, config.trainer.train_batch_size)

    # load the last checkpoint and resume if we want
    # TODO: add support for sharding
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
        # TODO: iter_data.seek(resume_step +1)
        for _ in range(resume_step + 1):
            next(iter_data)
        resume_step = resume_step + 1
    else:
        resume_step = 0

    # because we're running the model on each device in parallel, we need to make sure the model is on each device
    # (and the optimization state too)
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)

    # parallel training is fairly simple too. The body of the training loop should be a single function.
    # This function is being executed on each device in parallel
    @partial(pmap, axis_name="device")
    def train_step(model, opt_state, input_ids, targets, keys):
        def mean_loss_grad(model, x):
            loss, grads = compute_loss_and_grad(model, *x)
            loss = lax.pmean(loss, "device")
            grads = lax.pmean(grads, "device")
            return loss, grads

        loss, grads = accumulate_gradients(mean_loss_grad, model, (input_ids, targets, keys))

        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return loss, model, opt_state

    for step in range(resume_step, config.trainer.num_train_steps):
        time_in = time.perf_counter()
        micro_batch_shape = (len(devices), config.trainer.train_microbatches_per_step, config.trainer.per_device_train_batch_size)

        input_ids, targets = next(iter_data)
        micro_step_shape = (len(devices), config.trainer.train_microbatches_per_step, config.trainer.per_device_train_batch_size) + input_ids.shape[1:]
        input_ids = input_ids.reshape(micro_step_shape)
        targets = targets.reshape(micro_step_shape)
        my_key, training_key = jrandom.split(training_key, 2)
        micro_keys = shaped_rng_split(my_key, micro_batch_shape[:2])
        step_loss, model, opt_state = train_step(model, opt_state, input_ids, targets, micro_keys)
        step_loss = jnp.mean(step_loss).item()

        time_out = time.perf_counter()
        engine.run_hooks(StepInfo(step, model, opt_state, step_loss, training_key, time_out - time_in))

    last_step = StepInfo(config.trainer.num_train_steps, model, opt_state, step_loss, training_key, time_out - time_in)
    evaluate(last_step)
    save(last_step)


if __name__ == "__main__":
    # with jax.profiler.trace("logs"):
    main()
