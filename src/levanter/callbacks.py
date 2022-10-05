import copy
import logging
import time
from typing import Callable, Iterator, Optional, TypeVar

import jax.numpy as jnp
from tqdm import tqdm

import wandb
from levanter.checkpoint import save_checkpoint
from levanter.config import WandbConfig
from levanter.logging import log_optimizer_hyperparams, save_xla_dumps_to_wandb
from levanter.modeling_utils import RunningMean
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)


def save_model(run_dir, prepare_fn=None):
    if not prepare_fn:
        prepare_fn = lambda x: x  # noqa F731

    def save(info: StepInfo):
        # TODO: model sharded saving
        # TODO: when we do multi-machine model sharding we should do something cleverer.
        # it's actually pretty easy to save the model and the optimizer state
        # and enable resuming
        if info.step != 0:
            save_checkpoint(
                model=prepare_fn(info.model),
                training_state=((prepare_fn(info.opt_state)), info.next_key),
                step=info.step,
                checkpoint_path=f"{run_dir}/step-{info.step}",
            )

    return save


M = TypeVar("M")
X = TypeVar("X")
Y = TypeVar("Y")


def compute_validation_loss(
    loss_fn: Callable,  # [[M, ...], jax.numpy.ndarray],
    dataloader: Callable[[], Iterator[tuple]],
):
    def compute_loss(info: StepInfo):
        total_loss = RunningMean(shape=1)
        test_loader = dataloader()

        pbar = tqdm(test_loader, desc="eval", position=1, leave=False)
        for batch in pbar:
            loss = loss_fn(info.model, *batch)
            # this mean is over the devices, somewhat confusingly
            loss = jnp.mean(loss)
            total_loss.update(loss)
            pbar.set_postfix(loss=total_loss.mean.item())

        mean_loss = total_loss.mean.item()
        if wandb.run is not None:
            wandb.log({"eval/loss": mean_loss}, step=info.step)

        logger.info(f"validation loss: {mean_loss:.3f}")

        return total_loss

    return compute_loss


def log_to_wandb(step: StepInfo):
    wandb.log({"train/loss": step.loss, "global_step": step.step}, step=step.step)
    log_optimizer_hyperparams(step.opt_state, step=step.step)


def wandb_xla_logger(config: WandbConfig):
    last_mtime = wandb.run.start_time

    def log_xla_to_wandb(step: StepInfo):
        nonlocal last_mtime
        save_xla_dumps_to_wandb(last_mtime)
        # update time to now
        last_mtime = time.time()

    if config.save_xla_dumps:
        return log_xla_to_wandb
    else:
        return lambda x: None


def log_performance_stats(
    tokens_per_example: int,
    batch_size: int,
    flops_per_example: Optional[float] = None,
    prefix: Optional[str] = "throughput",
):
    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    def log_performance_stats(step_info: StepInfo):
        if step_info.step_duration != 0.0:
            wandb.log(
                {
                    wrap_key("examples_per_second"): float(batch_size) / step_info.step_duration,
                    wrap_key("tokens_per_second"): float(tokens_per_example) / step_info.step_duration * batch_size,
                    wrap_key("duration"): step_info.step_duration,
                },
                step=step_info.step,
            )

            if flops_per_example is not None:
                wandb.log(
                    {
                        wrap_key("gflops_per_second"): flops_per_example / 1e9 / step_info.step_duration * batch_size,
                    },
                    step=step_info.step,
                )

    return log_performance_stats


def pbar_logger(iterable=None, desc="train", **tqdm_mkwargs):
    kwargs = copy.copy(tqdm_mkwargs)
    if "desc" not in kwargs:
        kwargs["desc"] = desc
    if "iterable" not in kwargs:
        kwargs["iterable"] = iterable
    pbar = tqdm(**kwargs)

    def update_pbar(step: StepInfo):
        pbar.update(step.step - pbar.n)
        pbar.set_postfix(loss=step.loss)

    return update_pbar
