from dataclasses import dataclass
from typing import Any, Literal, Sequence, Union

import jax
from jax import numpy as jnp
from jax._src.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

import haliax.nn
from haliax import NamedArray, is_named_array
from haliax.jax_utils import is_jax_array_like

import levanter.tracker
from levanter.callbacks import JitCallback, M, S
from levanter.tracker.histogram import Histogram
from levanter.trainer_state import InsideJitInfo, TrainerState
from levanter.utils import jax_utils


Target = Literal["grads", "params", "opt_state", "updates"]


@dataclass(frozen=True)
class WatchConfig:
    watch_targets: Union[Sequence[Target], Target] = ("grads", "params")
    include_norms: bool = True
    include_per_parameter_norms: bool = True
    include_histograms: bool = False
    split_scan_layers: bool = True

    interval: int = 10

    @property
    def is_enabled(self) -> bool:
        return len(self.watch_targets) > 0 and self.interval > 0 and (self.include_norms or self.include_histograms)

    def build(self) -> "WatchCallback":
        return WatchCallback(
            watch_targets=self.watch_targets,
            include_norms=self.include_norms,
            include_per_parameter_norms=self.include_per_parameter_norms,
            include_histogram=self.include_histograms,
            split_scan_layers=self.split_scan_layers,
        )


class WatchCallback(JitCallback[S, M, dict[str, jax.Array | Histogram]]):
    """
    A unified callback for watching various aspects of training (gradients, parameters, optimizer state, updates).
    This callback combines the functionality of GradWatchCallback, ParamWatchCallback, OptStateWatchCallback,
    and UpdatesWatchCallback into a single callback.

    Args:
        watch_targets (Union[Sequence[str], str]): What to watch. Can be a comma-separated string or list of strings.
            Valid targets are: 'grads', 'params', 'opt_state', 'updates'.

        include_norms (bool): Whether to include norms in the logging.
        include_histogram (bool): Whether to include histograms in the logging.
        split_scan_layers (bool): Whether to split the scan layers into separate histograms/norms.
    """

    def __init__(
        self,
        watch_targets: Union[Sequence[str], str] = ("grads", "params"),
        include_norms: bool = True,
        include_per_parameter_norms: bool = True,
        include_histogram: bool = False,
        split_scan_layers: bool = True,
    ):
        if isinstance(watch_targets, str):
            watch_targets = [t.strip() for t in watch_targets.split(",")]

        self.watch_targets = watch_targets
        self.include_norms = include_norms
        self.include_per_parameter_norms = include_per_parameter_norms
        self.include_histogram = include_histogram
        self.split_scan_layers = split_scan_layers

        # Validate watch targets
        valid_targets = {"grads", "params", "opt_state", "updates"}
        invalid_targets = set(watch_targets) - valid_targets
        if invalid_targets:
            raise ValueError(f"Invalid watch targets: {invalid_targets}. Valid targets are: {valid_targets}")

    def inside_step(self, state: TrainerState[M], inside_info: InsideJitInfo[M]) -> dict[str, jax.Array | Histogram]:
        to_log = {}

        for target in self.watch_targets:
            if target == "grads":
                stats = summary_statistics_for_tree(
                    "grad",
                    inside_info.grads,
                    self.split_scan_layers,
                    include_histogram=self.include_histogram,
                    include_norms=self.include_norms,
                    include_per_parameter_norms=self.include_per_parameter_norms,
                )
                to_log.update(stats)

            elif target == "params":
                stats = summary_statistics_for_tree(
                    "params",
                    state.trainable_model,
                    self.split_scan_layers,
                    include_histogram=self.include_histogram,
                    include_norms=self.include_norms,
                    include_per_parameter_norms=self.include_per_parameter_norms,
                )
                to_log.update(stats)

            elif target == "updates":
                stats = summary_statistics_for_tree(
                    "updates",
                    inside_info.updates,
                    self.split_scan_layers,
                    include_histogram=self.include_histogram,
                    include_norms=self.include_norms,
                    include_per_parameter_norms=self.include_per_parameter_norms,
                )
                to_log.update(stats)

            elif target == "opt_state":
                # Special handling for optimizer state
                leaves = jax.tree.leaves_with_path(state.opt_state, is_leaf=lambda m: isinstance(m, type(state.model)))
                for path, v in leaves:
                    if not isinstance(v, type(state.model)):
                        continue

                    name = self._munge_key_name(path)
                    name_to_log = f"opt_state/{name}" if name else "opt_state"
                    this_stats = summary_statistics_for_tree(
                        name_to_log,
                        v,
                        self.split_scan_layers,
                        include_histogram=self.include_histogram,
                        include_norms=self.include_norms,
                        include_per_parameter_norms=self.include_per_parameter_norms,
                    )
                    to_log.update(this_stats)

        return to_log

    def on_step(self, step_info: S, cb_info: dict[str, jax.Array | Histogram]):
        levanter.tracker.log(cb_info, step=int(step_info.step))

    # Optimizer states can have weird/arbitrary structures, but the states we care about
    # are PyTrees with the same class as our model parameters (e.g., NamedArray, jax arrays, etc.)
    # opt_state/inner_state/1/nu/ --> we want nu
    def _munge_key_name(self, path: Sequence[Any]) -> str:
        """Helper method to format optimizer state keys."""
        if not path:
            return ""
        path_elem = path[-1]
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
            out = out[1:]

        return out


def summary_statistics_for_tree(
    prefix: str,
    tree: Any,
    split_scan_layers: bool,
    *,
    include_histogram: bool = False,
    include_norms: bool = True,
    include_per_parameter_norms: bool = True,
) -> dict[str, jax.Array | Histogram]:
    """
    Computes the summary statistics for a tree of (named) arrays.

    This function is designed to allow you to emulate the behavior of Wandb's PyTorch-only built-in gradient logging,
    but also works for any PyTree. It computes the Froebinius norm of each array,
    and optionally the histogram as well.

    Args:
        prefix: The prefix to use for logging.
        tree: The tree of arrays to compute the summary statistics for.
        split_scan_layers: Whether to split the scan layers into separate histograms/norms. Recommended.
        include_norms: Whether to include norms of the gradients. This increases overhead significantly.
        include_histogram: Whether to include histograms of the gradients. This increases overhead significantly.
        include_per_parameter_norms: Whether to include per-parameter norms.

    Returns:
        A dictionary of summary statistics.

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
                        norms[f"{key_path}.{i}.{k}"] = v[i]

                for k, v in vmapped_hists.items():
                    for i in range(g.Block.size):
                        hists[f"{key_path}.{i}.{k}"] = jax.tree.map(lambda x: x[i] if is_jax_array_like(x) else x, v)

            elif isinstance(g, NamedArray):
                # TODO: add linalg.norm to Haliax
                if include_norms:
                    norms[key_path] = jnp.linalg.norm(g.array)
                if include_histogram:
                    hist = Histogram.from_named_array(g)
                    hists[key_path] = hist
            elif is_jax_array_like(g):
                if include_norms:
                    norms[key_path] = jnp.linalg.norm(g)

                if include_histogram:
                    with jax.named_scope(f"histogram({prefix}/{key_path})"):
                        hist = Histogram.from_array(g)
                        hists[key_path] = hist

        return norms, hists

    norms_to_log: dict[str, jax.Array] = {}
    hists_to_log: dict[str, Histogram] = {}

    _rec_log_magnitudes(norms_to_log, hists_to_log, None, tree)

    to_log: dict[str, jax.Array | Histogram] = {}

    total_norm = jnp.zeros((), jnp.float32)
    for key, value in norms_to_log.items():
        if include_per_parameter_norms:
            to_log[f"{prefix}/norm/{key}"] = value
        total_norm += value

    to_log[f"{prefix}/norm/total"] = total_norm

    for key, hist in hists_to_log.items():
        to_log[f"{prefix}/hist/{key}"] = hist

    return to_log
