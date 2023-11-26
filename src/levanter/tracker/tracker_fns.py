import typing
from contextlib import AbstractContextManager
from typing import Any, Literal, Optional

import jax

from levanter.tracker import CompositeTracker, Tracker
from levanter.tracker.tensorboard import TensorboardTracker
from levanter.tracker.wandb import WandbTracker
from levanter.utils.jax_utils import is_inside_jit


_global_tracker: Optional["Tracker"] = None


def log_metrics(metrics: dict[str, Any], *, step, commit: Optional[bool] = None):
    """
    Log metrics to the global tracker.

    Args
        metrics: Metrics to log
        step: Step to log at
        commit: Whether to commit the metrics. If None, uses the default for the tracker.
    """
    global _global_tracker
    if _global_tracker is None:
        raise RuntimeError("No global tracker set")

    if is_inside_jit():
        # we're inside a jit, so we need to log from the host
        if commit:
            raise ValueError("Cannot commit from inside jit")
        jit_log_metrics(metrics, step=step)
    else:
        # TODO: do we need to coerce to np here?
        _global_tracker.log(metrics, step=step)


def jit_log_metrics(metrics, *, step=None):
    """uses jax effect callback to log to wandb from the host"""
    jax.debug.callback(log_metrics, metrics, step=step)


def log_summary(metrics: dict[str, Any]):
    """
    Log summary metrics to the global tracker.

    :param metrics: Metrics to log
    """
    global _global_tracker
    if _global_tracker is None:
        raise RuntimeError("No global tracker set")
    _global_tracker.log_summary(metrics)


@typing.overload
def current_tracker() -> "Tracker":
    ...


@typing.overload
def current_tracker(tracker: "Tracker") -> typing.ContextManager:
    """Returns a context manager for setting the global tracker"""
    ...


def current_tracker(
    tracker: Optional[Tracker] = None,
) -> Tracker | typing.ContextManager:
    """
    Get or set the global tracker. Note that setting the global tracker is not thread-safe,
    and using a tracker from multiple threads is only supported if the tracker itself is thread-safe.

    Args
      tracker: If provided, returns a context manager that sets the global tracker to the provided tracker when used.

    Returns
        If no tracker is provided, returns the current global tracker.
        If a tracker is provided, returns a context manager that sets the global tracker to the provided tracker when used.

    Examples
        >>> from levanter.tracker import current_tracker, log_metrics
        >>> from levanter.tracker.wandb import WandbTracker
        >>> with current_tracker(WandbTracker()):
        ...     log_metrics({"foo": 1}, step=0)
        ...     current_tracker().log({"foo": 2}, step=1)
    """
    global _global_tracker
    if tracker is None:
        if _global_tracker is None:
            raise RuntimeError("No global tracker set")
        return _global_tracker
    else:
        return _GlobalLoggerContextManager(tracker)


@typing.overload
def get_tracker(name: Literal["wandb"]) -> WandbTracker:
    ...


@typing.overload
def get_tracker(name: Literal["tensorboard"]) -> TensorboardTracker:
    ...


@typing.overload
def get_tracker(name: str) -> Tracker:
    ...


def get_tracker(name: str) -> Tracker:
    """
    Lookup a tracker in the current global tracker with the provided name.

    :param name: Name of the tracker
    :return: The tracker
    """
    tracker = current_tracker()
    if isinstance(tracker, CompositeTracker):
        for t in tracker.loggers:
            if t.name == name:
                return t
    elif tracker.name == name:
        return tracker

    raise KeyError(f"Tracker with name {name} not found")


class _GlobalLoggerContextManager(AbstractContextManager):
    def __init__(self, tracker: "Tracker"):
        self.tracker = tracker

    def __enter__(self):
        global _global_tracker
        self.old_tracker = _global_tracker
        _global_tracker = self.tracker

        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_tracker
        _global_tracker = self.old_tracker
