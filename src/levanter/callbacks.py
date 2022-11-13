import copy
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from typing import Callable, Iterator, Optional, TypeVar

import humanfriendly
import jax
import jax.numpy as jnp
from tqdm import tqdm

import wandb
from levanter.config import WandbConfig
from levanter.logging import log_optimizer_hyperparams, save_xla_dumps_to_wandb
from levanter.modeling_utils import RunningMean
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)

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

        # log these totals because it's useful for comparing different seqlens, batch sizes, etc
        total_tokens = tokens_per_example * batch_size * step_info.step
        wandb.log({wrap_key("total_tokens"): total_tokens}, step=step_info.step)

        if flops_per_example:
            total_flops = flops_per_example * batch_size * step_info.step
            wandb.log({wrap_key("total_gflops"): total_flops / 1e9}, step=step_info.step)

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


def log_memory_usage(sample_interval: float = 1.0):
    """
    Logs memory usage to wandb. This runs a loop that samples memory usage every `sample_interval` seconds.
    We only log when hooks are invoked, so there's not much point in running this much more frequently than you invoke
    the hook.

    I think it's a good idea to run this in a separate thread, so that you sample from random points, but I'm not sure.
    :param sample_interval:
    :return:
    """

    directory = "/dev/shm"
    # macos doesn't have /dev/shm
    if not os.path.exists(directory):
        directory = tempfile.gettempdir()

    # a lot of this code is lifted from https://github.com/ayaka14732/jax-sm CC-0

    def inner():
        import posix
        import time

        while True:
            jax.profiler.save_device_memory_profile(f"{directory}/memory.prof.new")
            posix.rename(f"{directory}/memory.prof.new", f"{directory}/memory.prof")
            time.sleep(sample_interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()

    def log_memory_usage(step: StepInfo):
        output = subprocess.run(
            args=f"go tool pprof -tags {directory}/memory.prof".split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.decode("utf-8")

        # output looks like this:
        #          2.4MB (12.53%): TFRT_CPU_0
        #          2.4MB (12.50%): TFRT_CPU_1
        #          2.4MB (12.50%): TFRT_CPU_2
        #          2.4MB (12.50%): TFRT_CPU_3
        #          2.4MB (12.50%): TFRT_CPU_4
        #          2.4MB (12.50%): TFRT_CPU_5
        #          2.4MB (12.50%): TFRT_CPU_6
        #          2.4MB (12.50%): TFRT_CPU_7
        #
        #  kind: Total 19.5MB
        #         18.9MB (97.20%): buffer
        #        558.4kB ( 2.80%): executable
        per_device, by_kind = output.split("kind: Total ")

        # first, get the total memory usage
        regex = re.compile(r"^(\d+\.\d+[a-zA-Z]+)")
        match = regex.search(by_kind)
        if match:
            memory_usage = humanfriendly.parse_size(match.group(1))
            wandb.log({"memory/total": memory_usage / 1e6}, step=step.step)

        # now, get the memory usage per device.
        # split the output at kind: Total
        regex = re.compile(r"([\d\.]+[a-zA-Z]+) \(([\d\.]+)%\): ([a-zA-Z0-9_]+)")
        for match in regex.finditer(per_device):
            memory_usage = humanfriendly.parse_size(match.group(1))
            device_name = match.group(3)
            wandb.log({f"memory/device/{device_name}": memory_usage / 1e6}, step=step.step)

        # now, get the memory usage per kind.
        # same regex as above
        for match in regex.finditer(by_kind):
            memory_usage = match.group(1)
            memory_usage = humanfriendly.parse_size(memory_usage)
            wandb.log({f"memory/{match.group(3)}": memory_usage / 1e6}, step=step.step)

    return log_memory_usage


# from https://stackoverflow.com/questions/42865724/parse-human-readable-filesizes-into-bytes
units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def _parse_size(size):
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])
