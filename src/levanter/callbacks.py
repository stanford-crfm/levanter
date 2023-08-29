import copy
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import warnings
from typing import Callable, Iterable, Optional

import humanfriendly
import jax
import wandb
from tqdm import tqdm

from levanter.logging import WandbConfig, log_optimizer_hyperparams, save_xla_dumps_to_wandb
from levanter.trainer import StepInfo
from levanter.utils.jax_utils import jnp_to_python
from levanter.visualization import compute_and_visualize_log_probs as viz_probs


logger = logging.getLogger(__name__)


def eval_loss_loop(loss_fn, model, dataset, max_batches: Optional[int] = None):
    total_loss = 0.0
    n = 0

    pbar = tqdm(dataset, desc="eval", position=1, leave=False)
    for batch in pbar:
        loss = loss_fn(model, batch)
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=total_loss / n)

        if max_batches is not None and n >= max_batches:
            break

    if n > 0:
        total_loss /= n

    return total_loss


def compute_validation_loss(
    loss_fn: Callable,  # [[M, ...], jax.numpy.ndarray],
    dataset: Iterable,
    max_batches: Optional[int] = None,
):
    def compute_loss(info: StepInfo):
        loss = eval_loss_loop(loss_fn, info.model, dataset, max_batches=max_batches)

        if wandb.run is not None:
            wandb.log({"eval/loss": loss}, step=info.step)

        logger.info(f"validation loss: {loss:.3f}")

        return loss

    return compute_loss


def log_to_wandb(step: StepInfo):
    wandb.log({"train/loss": step.loss, "global_step": step.step}, step=step.step)
    log_optimizer_hyperparams(step.opt_state, step=step.step)


def wandb_xla_logger(config: WandbConfig):
    last_mtime = wandb.run and wandb.run.start_time or time.time()

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
        pbar.update(step.next_step - pbar.n)
        pbar.set_postfix(loss=jnp_to_python(step.loss))

    return update_pbar


def log_memory_usage(sample_interval: float = 1.0, log_individual_devices: bool = False):
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

    tempfile_name = os.path.join(directory, f"memory_usage_{os.getpid()}.prof")

    # a lot of this code is lifted from https://github.com/ayaka14732/jax-smi CC-0

    def inner():
        import posix
        import time

        while True:
            jax.profiler.save_device_memory_profile(f"{tempfile_name}.new")
            posix.rename(f"{tempfile_name}.new", tempfile_name)
            time.sleep(sample_interval)

    thread = threading.Thread(target=inner, daemon=True)
    thread.start()

    def log_memory_usage(step: StepInfo):
        process = subprocess.run(
            args=f"go tool pprof -tags {tempfile_name}".split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        if process.returncode != 0:
            warnings.warn("failed to run pprof. Is go installed?")
            return

        output = process.stdout.decode("utf-8")

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

        # gpus look like this:
        #          1.0MB ( 0.00%): gpu:0
        per_device, by_kind = output.split("kind: Total ")

        # first, get the total memory usage
        regex = re.compile(r"^(\d+\.\d+[a-zA-Z]+)")
        match = regex.search(by_kind)
        if match:
            memory_usage = humanfriendly.parse_size(match.group(1))
            wandb.log({"memory/total": memory_usage / 1e6}, step=step.step)

        # this works for the "kind" and the individual devices
        regex = re.compile(r"([\d.]+[a-zA-Z]+) \(([\d.]+)%\): ([\w\d:_]+)")

        if log_individual_devices:
            # now, get the memory usage per device.
            # split the output at kind: Total
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


def compute_and_visualize_log_probs(test_data, tokenizer, log_prob_fn, html_dir: str, max_docs=128):
    """
        Computes log probabilities for a dataset and visualizes them using visdom.

        Args:
            test_data (Type): The test dataset for computation. Specify the type expected.
            tokenizer (Type): The tokenizer to be used. Specify the type expected.
            log_prob_fn (function): A function that takes a model and a batch; then returns the log probabilities for each token.
            html_dir (str): The directory where the HTML output will be written.
            max_docs (int): The maximum number of documents to process.

        Returns:
    function: A function that takes a step info and computes and visualizes the log probabilities.
    """

    def compute_and_viz_log_probs(step: StepInfo):
        model = step.model
        os.makedirs(html_dir, exist_ok=True)
        path = os.path.join(html_dir, f"step_{step}.html")

        viz_probs(path, model, tokenizer, log_prob_fn, test_data, max_docs=max_docs)
        wandb.log({"log_probs": wandb.Html(path)}, step=step.step)

    return compute_and_viz_log_probs
