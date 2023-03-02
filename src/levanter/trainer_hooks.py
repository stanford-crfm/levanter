from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

from chex import PRNGKey
from jaxtyping import PyTree


S = TypeVar("S")  # State
B = TypeVar("B", covariant=True)  # Batch
Aux = TypeVar("Aux", covariant=True)  # Auxiliary per-iteration results
T = TypeVar("T")


@dataclass
class StepInfo:
    step: int
    model: PyTree
    opt_state: Any
    loss: float
    next_key: PRNGKey
    step_duration: float


class TrainerHooks:
    hooks: List[Callable[[StepInfo], None]] = []

    def run_hooks(self, info: StepInfo):
        for hook in self.hooks:
            hook(info)

    def add_hook(self, fn: Optional[Callable[[StepInfo], Any]] = None, *, every: int = 1):
        def decorator(fn: Callable[[StepInfo], None]):
            if every == 1:
                self.hooks.append(fn)
            else:
                self.hooks.append(lambda info: fn(info) if info.step % every == 0 else None)
            return fn

        if fn is None:
            return decorator
        else:
            return decorator(fn)
