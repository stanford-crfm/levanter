from typing import Any, Sequence

import jax
from jax import numpy as jnp
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

import haliax.nn
from haliax import NamedArray, is_named_array
from haliax.jax_utils import is_jax_array_like

import levanter.tracker
from levanter.callbacks import JitCallback, M, S, StepInfo
from levanter.tracker.histogram import Histogram
from levanter.trainer_state import InsideJitInfo, TrainerState
from levanter.utils import jax_utils


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

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]):
        return summary_statistics_for_tree(
            self.prefix, inside_info.grads, self.split_scan_layers, self.include_histogram
        )

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

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]):
        return summary_statistics_for_tree(
            self.prefix, state.trainable_model, self.split_scan_layers, self.include_histogram
        )

    def on_step(self, step_info: StepInfo[S], cb_info: dict[str, float | Histogram]):
        levanter.tracker.log(cb_info, step=step_info.step)


class OptStateWatchCallback(JitCallback[S, M, dict[str, float | Histogram]]):
    """
    Same as [levanter.callbacks.GradWatchCallback][], but for the optimizer state.

    Args:
        prefix (str): The prefix to use for logging.
        include_histogram (bool): Whether to include histograms of the gradients.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms
    """

    def __init__(
        self,
        prefix: str = "opt_state",
        include_histogram: bool = True,
        split_scan_layers: bool = True,
    ):
        self.prefix = prefix
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]):
        to_return = {}
        # return summary_statistics_for_tree(
        #     self.prefix, state.opt_state, self.split_scan_layers, self.include_histogram
        # )

        leaves = jax.tree.leaves_with_path(
            state.opt_state, is_leaf=lambda m: isinstance(m, type(state.model))
        )  # noqa: E731
        for path, v in leaves:
            if not isinstance(v, type(state.model)):
                continue

            name = self._munge_key_name(path)

            if name:
                name_to_log = f"{self.prefix}/{name}"
            else:
                name_to_log = self.prefix

            this_hists = summary_statistics_for_tree(name_to_log, v, self.split_scan_layers, self.include_histogram)

            to_return.update(this_hists)

        return to_return

    def on_step(self, step_info: StepInfo[S], cb_info: dict[str, float | Histogram]):
        levanter.tracker.log(cb_info, step=step_info.step)

    # Optimizer states can have weird/arbitrary structures, but the states we care about
    # are PyTrees with the same class as our model parameters (e.g., NamedArray, jax arrays, etc.)
    # opt_state/inner_state/1/nu/ --> we want nu
    def _munge_key_name(self, path: Sequence[Any]) -> str:
        if not path:
            return ""
        path_elem = path[-1]  # Get the last element in the path, which is the actual value
        match path_elem:
            case SequenceKey(idx):  # type: ignore
                out = f"{idx}"
            case DictKey(key):  # type: ignore
                out = f"{key}"
            case GetAttrKey():  # type: ignore
                out = str(path_elem)
            case FlattenedIndexKey(idx):  # type: ignore
                out = f"{idx}"
            case _:
                path_elem = str(path_elem)
                out = f"{path_elem}"

        if out.startswith("."):
            out = out[1:]  # Remove leading dot if it exists

        return out


class UpdatesWatchCallback(JitCallback[S, M, dict[str, float | Histogram]]):
    """
    Same as [levanter.callbacks.GradWatchCallback][], but for the updates to the model parameters.

    Args:
        prefix (str): The prefix to use for logging.
        include_histogram (bool): Whether to include histograms of the updates.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms
    """

    def __init__(
        self,
        prefix: str = "updates",
        include_histogram: bool = True,
        split_scan_layers: bool = True,
    ):
        self.prefix = prefix
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]):
        return summary_statistics_for_tree(
            self.prefix, inside_info.updates, self.split_scan_layers, self.include_histogram
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
                    with jax.named_scope(f"histogram({prefix}/{key_path})"):
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
