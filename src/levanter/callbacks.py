import copy
import logging as pylogging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Callable, Optional

import humanfriendly
import jax
from tqdm_loggable import tqdm_logging
from tqdm_loggable.auto import tqdm

import levanter.tracker
from levanter.data import AsyncDataset, DataLoader
from levanter.logging import save_xla_dumps_to_wandb
from levanter.tracker.helpers import log_optimizer_hyperparams
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import StepInfo
from levanter.utils import flop_utils
from levanter.utils.jax_utils import barrier_sync, jnp_to_python
from levanter.visualization import compute_and_visualize_log_probs as viz_probs


logger = pylogging.getLogger(__name__)


def log_epoch_progress(total_tokens_future, tokens_per_example, batch_size, max_epochs: Optional[int] = None):
    total_tokens = None

    def log_epoch(step_info: StepInfo):
        nonlocal total_tokens
        if total_tokens is None:
            if not total_tokens_future.done():
                if step_info.step % 1000 == 0:
                    logger.info("Dataset not finished. Can't compute epochs.")
                return  # We don't have the total tokens yet, so we can't calculate epoch
            total_tokens = total_tokens_future.result()

        # Get the total processed tokens from the metrics logged by log_performance_stats
        processed_tokens = tokens_per_example * batch_size * step_info.step

        # If we're doing multiple epochs, adjust the denominator
        total_tokens_for_epochs = total_tokens * max_epochs if max_epochs else total_tokens
        current_epoch = processed_tokens / total_tokens_for_epochs

        levanter.tracker.log_metrics({"train/current_epoch": current_epoch}, step=step_info.step)

    return log_epoch


def get_total_dataset_tokens(ds: AsyncDataset, seq_length: int):
    def log_length():
        # If ds.async_len() is the only option, run it in an event loop inside the thread
        import asyncio

        async def compute_length():
            length = await ds.dataset.async_len()
            return length

        # Run the async function synchronously in this thread
        length = asyncio.run(compute_length())
        total_tokens = length * seq_length
        levanter.tracker.log_summary({"dataset/total_tokens": total_tokens})
        return total_tokens

    # Create a ThreadPoolExecutor with a single worker thread
    executor = ThreadPoolExecutor(max_workers=1)
    # Submit the log_length function to be executed in a separate thread
    future = executor.submit(log_length)
    return future


def eval_loss_loop(loss_fn, model, dataset, max_batches: Optional[int] = None, name: Optional[str] = None):
    total_loss = 0.0
    total_load_time = 0.0
    total_loss_time = 0.0
    n = 0

    if name is not None:
        desc = f"eval {name}"
    else:
        desc = "eval"

    _tqdm_logging_one_time_setup()
    pbar = tqdm(dataset, desc=desc, position=1, leave=False, total=max_batches)

    iter_ = iter(pbar)
    while True:
        time_in = time.time()
        batch = next(iter_, None)
        if batch is None:
            break
        load_time = time.time() - time_in
        total_load_time += load_time
        loss = loss_fn(model, batch)
        total_loss += loss.item()
        n += 1
        loss_time = time.time() - time_in - load_time
        total_loss_time += loss_time

        pbar.set_postfix(loss=total_loss / n)

        if max_batches is not None and n >= max_batches:
            break

    if n > 0:
        total_loss /= n

    # logger.info(f"eval loading time: {total_load_time / n:.3f} s/ba")
    # logger.info(f"eval loss time: {total_loss_time / n:.3f} s/ba")

    return total_loss


def compute_validation_loss(
    loss_fn: Callable,  # [[M, ...], jax.numpy.ndarray],
    dataset: DataLoader,
    max_batches: Optional[int] = None,
    name: Optional[str] = None,
):
    def compute_loss(info: StepInfo):
        loss = eval_loss_loop(loss_fn, info.model, dataset, max_batches=max_batches, name=name)

        prefix = "eval"
        if name:
            prefix += "/" + name
        levanter.tracker.log_metrics({f"{prefix}/loss": loss}, step=info.step)

        if name:
            logger.info(f"{name} validation loss: {loss:.3f}")
        else:
            logger.info(f"validation loss: {loss:.3f}")

        return loss

    return compute_loss


def log_step_info(total_steps: Optional[int]):
    def log_step_info_inner(step: StepInfo):
        metrics = {"train/loss": step.loss, "global_step": step.step}
        if total_steps:
            metrics["run_progress"] = step.step / total_steps
        log_optimizer_hyperparams(step.opt_state, step=step.step, prefix="optim")
        levanter.tracker.log_metrics(metrics, step=step.step)

    return log_step_info_inner


def wandb_xla_logger(config: WandbConfig):
    import wandb

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

    device_count = jax.device_count()
    device = jax.devices()[0]

    flops_per_device = flop_utils.device_hardware_flops(device)

    if flops_per_device is not None:
        theoretical_flops = flops_per_device * device_count

        levanter.tracker.log_summary(
            {
                wrap_key("device_kind"): device.device_kind,
                wrap_key("theoretical_flops_per_device"): flops_per_device,
                wrap_key("theoretical_flops"): theoretical_flops,
            }
        )

    if flops_per_example is not None:
        levanter.tracker.log_summary({wrap_key("flops_per_example"): flops_per_example})

    def log_performance_stats(step_info: StepInfo):
        dict_to_log = {}

        # log these totals because it's useful for comparing different seqlens, batch sizes, etc
        total_tokens = tokens_per_example * batch_size * step_info.step
        dict_to_log["total_tokens"] = total_tokens

        if flops_per_example:
            total_flops = flops_per_example * batch_size * step_info.step
            dict_to_log["total_gflops"] = total_flops / 1e9

        if step_info.step_duration != 0.0:
            dict_to_log["examples_per_second"] = float(batch_size) / step_info.step_duration
            dict_to_log["tokens_per_second"] = float(tokens_per_example) / step_info.step_duration * batch_size
            dict_to_log["duration"] = step_info.step_duration

            if flops_per_example is not None:
                model_flops_instant = flops_per_example / step_info.step_duration * batch_size
                dict_to_log["gflops_per_second"] = model_flops_instant / 1e9

                if flops_per_device is not None:
                    mfu_instant = model_flops_instant / theoretical_flops * 100.0
                    dict_to_log["mfu"] = mfu_instant

        dict_to_log = {wrap_key(k): v for k, v in dict_to_log.items()}
        levanter.tracker.log_metrics(dict_to_log, step=step_info.step)

    return log_performance_stats


def pbar_logger(iterable=None, desc="train", **tqdm_mkwargs):
    kwargs = copy.copy(tqdm_mkwargs)
    if "desc" not in kwargs:
        kwargs["desc"] = desc
    if "iterable" not in kwargs:
        kwargs["iterable"] = iterable

    _tqdm_logging_one_time_setup()
    pbar = tqdm(**kwargs)

    def update_pbar(step: StepInfo):
        pbar.update(step.next_step - pbar.n)
        pbar.set_postfix(loss=jnp_to_python(step.loss))

    return update_pbar


def log_memory_usage(sample_interval: float = 1.0, log_individual_devices: bool = False):
    """
    Logs memory usage. This runs a loop that samples memory usage every `sample_interval` seconds.
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
            levanter.tracker.log_metrics({"memory/total": memory_usage / 1e6}, step=step.step)

        # this works for the "kind" and the individual devices
        regex = re.compile(r"([\d.]+[a-zA-Z]+) \(([\d.]+)%\): ([\w\d:_]+)")

        if log_individual_devices:
            # now, get the memory usage per device.
            # split the output at kind: Total
            for match in regex.finditer(per_device):
                memory_usage = humanfriendly.parse_size(match.group(1))
                device_name = match.group(3)
                levanter.tracker.log_metrics({f"memory/device/{device_name}": memory_usage / 1e6}, step=step.step)

        # now, get the memory usage per kind.
        # same regex as above
        for match in regex.finditer(by_kind):
            memory_usage = match.group(1)
            memory_usage = humanfriendly.parse_size(memory_usage)
            levanter.tracker.log_metrics({f"memory/{match.group(3)}": memory_usage / 1e6}, step=step.step)

    return log_memory_usage


def profile(path: str, start_step: int, num_steps: int, create_perfetto_link: bool) -> Callable[[StepInfo], None]:
    print(f"create_perfetto_link: {create_perfetto_link}")

    def profiler_callback_fn(step: StepInfo):
        # -1 b/c step is the finished step
        if step.step == start_step - 1:
            _create_perfetto_link = create_perfetto_link and jax.process_index() == 0
            logger.info(f"Starting profiler until step {start_step + num_steps}.")
            jax.profiler.start_trace(path, create_perfetto_link=_create_perfetto_link, create_perfetto_trace=True)
        elif step.step == start_step + num_steps - 1:
            if create_perfetto_link:
                logger.info(
                    f"Stopping profiler. Process 0 will open a perfetto link. I am process {jax.process_index()}"
                )
            else:
                logger.info("Stopping profiler.")
            # so, annoyingly, gcloud ssh doesn't reliably flush stdout here, so we need to spin up
            # a thread to flush and print periodically until we make it past stop_trace
            # (note: stop_trace blocks if perfetto is enabled)
            event = threading.Event()
            if create_perfetto_link and jax.process_index() == 0:
                _flush_while_waiting(event)

            jax.profiler.stop_trace()

            if create_perfetto_link and jax.process_index() == 0:
                event.set()

            levanter.tracker.current_tracker().log_artifact(path, type="jax_profile")
            barrier_sync()

    return profiler_callback_fn


def _flush_while_waiting(event):
    def flush_stdout():
        sys.stdout.flush()
        time.sleep(5)
        while not event.is_set():
            print("Waiting...", flush=True)
            time.sleep(5)

    thread = threading.Thread(target=flush_stdout)
    thread.start()


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
        path = os.path.join(html_dir, f"step_{step.step}.html")

        viz_probs(path, model, tokenizer, log_prob_fn, test_data, max_docs=max_docs)
        # TODO: convert to generic logging
        import wandb

        wandb.log({"log_probs": wandb.Html(path)}, step=step.step)

    return compute_and_viz_log_probs


_did_tqdm_logging_one_time_setup = False


def _tqdm_logging_one_time_setup():
    global _did_tqdm_logging_one_time_setup
    if _did_tqdm_logging_one_time_setup:
        return
    _did_tqdm_logging_one_time_setup = True
    tqdm_logging.tqdm_logging.set_log_rate(timedelta(seconds=60))
