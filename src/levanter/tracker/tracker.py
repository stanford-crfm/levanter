import abc
import typing
from contextlib import AbstractContextManager
from typing import Any, List, Optional

import draccus
import jax

from levanter.utils.jax_utils import is_inside_jit


class Tracker(abc.ABC):
    """
    A tracker is responsible for logging metrics, hyperparameters, and artifacts.
    Meant to be used with the [current_tracker][] context manager, but can also be used directly.

    The name is borrowed from Accelerate.

    Examples:
        >>> from levanter.tracker import current_tracker, log_metrics
        >>> from levanter.tracker.wandb import WandbTracker
        >>> with current_tracker(WandbTracker()):
        ...     log_metrics({"foo": 1}, step=0)
    """

    @abc.abstractmethod
    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    @abc.abstractmethod
    def log(self, metrics: dict[str, typing.Any], *, step, commit: Optional[bool] = None):
        """
        Log metrics to the tracker. Step is always required.

        Args:
            metrics: Metrics to log
            step: Step to log at
            commit: Whether to commit the metrics. If None, uses the default for the tracker.
        """
        pass

    @abc.abstractmethod
    def log_summary(self, metrics: dict[str, Any]):
        pass

    @abc.abstractmethod
    def log_artifact(self, artifact, *, name: Optional[str] = None, type: Optional[str] = None):
        pass


class CompositeTracker(Tracker):
    def __init__(self, loggers: List[Tracker]):
        self.loggers = loggers

    def log_hyperparameters(self, hparams: dict[str, Any]):
        for tracker in self.loggers:
            tracker.log_hyperparameters(hparams)

    def log(self, metrics: dict[str, Any], *, step, commit=None):
        for tracker in self.loggers:
            tracker.log(metrics, step=step, commit=commit)

    def log_summary(self, metrics: dict[str, Any]):
        for tracker in self.loggers:
            tracker.log_summary(metrics)

    def log_artifact(self, artifact, *, name: Optional[str] = None, type: Optional[str] = None):
        for tracker in self.loggers:
            tracker.log_artifact(artifact, name=name, type=type)


_global_tracker: Optional["Tracker"] = None


class TrackerConfig(draccus.PluginRegistry):
    discover_packages_path = "levanter.tracker"

    def init(self, run_id: Optional[str], hparams=None) -> Tracker:
        raise NotImplementedError


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
    Get or set the global tracker.

    :param tracker: If provided, returns a context manager that sets the global tracker to the provided tracker when used.
    :return: The global tracker, or a context manager for setting the global tracker.
    """
    global _global_tracker
    if tracker is None:
        if _global_tracker is None:
            raise RuntimeError("No global tracker set")
        return _global_tracker
    else:
        return _GlobalLoggerContextManager(tracker)


class _GlobalLoggerContextManager(AbstractContextManager):
    def __init__(self, tracker: "Tracker"):
        self.tracker = tracker

    def __enter__(self):
        global _global_tracker
        self.old_tracker = _global_tracker
        _global_tracker = self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_tracker
        _global_tracker = self.old_tracker


class NullTracker(Tracker):
    def log_hyperparameters(self, hparams: dict[str, Any]):
        pass

    def log(self, metrics: dict[str, Any], *, step, commit: Optional[bool] = None):
        pass

    def log_summary(self, metrics: dict[str, Any]):
        pass

    def log_artifact(self, artifact, *, name: Optional[str] = None, type: Optional[str] = None):
        pass


class NullTrackerConfig(TrackerConfig):
    def init(self, run_id: Optional[str], hparams=None) -> Tracker:
        return NullTracker()
