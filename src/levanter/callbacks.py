import abc
import copy
import logging as pylogging
import os
import sys
import threading
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Generic, Optional, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from tqdm_loggable import tqdm_logging
from tqdm_loggable.auto import tqdm

import haliax.nn
from haliax import NamedArray, is_named_array
from haliax.jax_utils import is_jax_array_like

import levanter.tracker
from levanter.data import AsyncDataset, DataLoader
from levanter.tracker.helpers import log_optimizer_hyperparams
from levanter.tracker.histogram import Histogram
from levanter.tracker.wandb import WandbConfig
from levanter.trainer_state import TrainerState
from levanter.utils import flop_utils, jax_utils
from levanter.utils.jax_utils import barrier_sync, jnp_to_python
from levanter.utils.logging import save_xla_dumps_to_wandb
from levanter.visualization import compute_and_visualize_log_probs as viz_probs


logger = pylogging.getLogger(__name__)

M = TypeVar("M")  # Model
M_con = TypeVar("M_con", bound=PyTree, contravariant=True)
S = TypeVar("S", bound=TrainerState)
CBInfo = TypeVar("CBInfo")


@dataclass
class StepInfo(Generic[S]):
    """
    Information about a step that was just completed. This includes the trainer state, the loss, and the duration of the
    step.

    Note that the step is 0-indexed, so if you want the next step, use `next_step`.
    """

    state: S
    loss: float
    step_duration: float

    model = property(lambda self: self.state.model)
    opt_state = property(lambda self: self.state.opt_state)

    step = property(lambda self: int(self.state.step) - 1)
    """
    The step that was just completed. If you want the next step, use `next_step`.
    """

    next_step = property(lambda self: int(self.state.step))


class Callback(ABC, Generic[S]):
    """
    A callback that can be called at the end of a step. This is useful for logging, profiling, and other side effects.
    """

    @abc.abstractmethod
    def on_step(self, info: StepInfo[S], force: bool = False):
        ...


class LambdaCallback(Callback[S]):
    def __init__(self, fn: Callable[[StepInfo[S]], Any]):
        self.fn = fn

    def on_step(self, info: StepInfo[S], force: bool = False):
        self.fn(info)


class JitCallback(ABC, Generic[S, M, CBInfo]):
    """
    A callback that gets called in two phases: inside the step (inside jit), and after the step (outside jit).
    You have access to the gradients inside the step, so you can compute statistics on them.
    """

    @abc.abstractmethod
    def inside_step(self, state: S, grad: M) -> CBInfo:
        ...

    @abc.abstractmethod
    def on_step(self, step_info: S, cb_info: CBInfo):
        ...


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

        levanter.tracker.log({"train/current_epoch": current_epoch}, step=step_info.step)

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
        levanter.tracker.log({f"{prefix}/loss": loss}, step=info.step)

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
        levanter.tracker.log(metrics, step=step.step)

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
        levanter.tracker.log(dict_to_log, step=step_info.step)

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


def profile(path: str, start_step: int, num_steps: int, create_perfetto_link: bool) -> Callable[[StepInfo], None]:
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


class GradWatchCallback(JitCallback[S, M, dict[str, float | Histogram]]):
    """
    Emulates the behavior of Wandb's PyTorch-only built-in gradient logging (wandb.watch)

    Args:
        prefix (str): The prefix to use for logging.
        include_histogram (bool): Whether to include histograms of the gradients.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms
    """

    def __init__(
        self,
        prefix: str = "grad",
        include_histogram: bool = True,
        split_scan_layers: bool = True,
    ):
        self.prefix = prefix
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

    def inside_step(self, state: TrainerState[M], grad: M):
        return summary_statistics_for_tree(self.prefix, grad, self.split_scan_layers, self.include_histogram)

    def on_step(self, step_info: StepInfo[S], cb_info: dict[str, float | Histogram]):
        levanter.tracker.log(cb_info, step=step_info.step)


class ParamWatchCallback(JitCallback[S, M, dict[str, float | Histogram]]):
    """
    Emulates the behavior of Wandb's PyTorch-only built-in gradient logging (wandb.watch)

    Args:
        prefix (str): The prefix to use for logging.
        include_histogram (bool): Whether to include histograms of the gradients.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms
    """

    def __init__(
        self,
        prefix: str = "params",
        include_histogram: bool = True,
        split_scan_layers: bool = True,
    ):
        self.prefix = prefix
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

    def inside_step(self, state: TrainerState[M], grad: M):
        return summary_statistics_for_tree(
            self.prefix, state.trainable_model, self.split_scan_layers, self.include_histogram
        )

    def on_step(self, step_info: StepInfo[S], cb_info: dict[str, float | Histogram]):
        levanter.tracker.log(cb_info, step=step_info.step)


def summary_statistics_for_tree(
    prefix: str, tree: M, split_scan_layers: bool, include_histogram: bool
) -> dict[str, float | Histogram]:
    """
    Computes the summary statistics for a tree of (named) arrays.

    This function is designed to allow you to emulate the behavior of Wandb's PyTorch-only built-in gradient logging,
    but also works for any PyTree. It computes the Froebinius norm of each array,
    and optionally the histogram as well.

    Args:
        prefix: The prefix to use for logging.
        tree: The tree of arrays to compute the summary statistics for.
        split_scan_layers: Whether to split the scan layers into separate histograms/norms. Recommended.
        include_histogram: Whether to include histograms of the gradients. This increases overhead significantly.

    Returns:

    """
    if split_scan_layers:
        is_leaf = lambda n: isinstance(n, haliax.nn.Stacked) or is_named_array(n)  # noqa: E731
    else:
        is_leaf = is_named_array

    def _rec_log_magnitudes(norms, hists, path_prefix, tree):
        leaf_key_paths = jax_utils.leaf_key_paths(tree, prefix=path_prefix, is_leaf=is_leaf)
        del path_prefix
        for key_path, g in zip(
            jax.tree.leaves(leaf_key_paths, is_leaf=is_leaf),
            jax.tree.leaves(tree, is_leaf=is_leaf),
            strict=True,
        ):
            if split_scan_layers and isinstance(g, haliax.nn.Stacked):
                vmapped_norms, vmapped_hists = haliax.vmap(_rec_log_magnitudes, g.Block)({}, {}, "", g.stacked)

                for k, v in vmapped_norms.items():
                    for i in range(g.Block.size):
                        norms[f"{key_path}/{i}/{k}"] = v[i]

                for k, v in vmapped_hists.items():
                    for i in range(g.Block.size):
                        hists[f"{key_path}/{i}/{k}"] = jax.tree.map(lambda x: x[i] if is_jax_array_like(x) else x, v)

            elif isinstance(g, NamedArray):
                # TODO: add linalg.norm to Haliax
                norms[key_path] = jnp.linalg.norm(g.array)
                if include_histogram:
                    hist = Histogram.from_named_array(g)
                    hists[key_path] = hist
            elif is_jax_array_like(g):
                norms[key_path] = jnp.linalg.norm(g)

                if include_histogram:
                    hist = Histogram.from_array(g)
                    hists[key_path] = hist

        return norms, hists

    norms_to_log: dict[str, jax.Array] = {}
    hists_to_log: dict[str, Histogram] = {}

    _rec_log_magnitudes(norms_to_log, hists_to_log, None, tree)

    to_log: dict = {}

    for key, value in norms_to_log.items():
        to_log[f"{prefix}/norm/{key}"] = value

    for key, value in hists_to_log.items():
        to_log[f"{prefix}/hist/{key}"] = value

    return to_log
