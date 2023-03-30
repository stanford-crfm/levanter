from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from chex import PRNGKey
from jaxtyping import PyTree


@dataclass
class StepInfo:
    step: int
    model: PyTree
    opt_state: Any
    loss: float
    next_key: PRNGKey
    step_duration: float


@dataclass
class _Hook:
    fn: Callable[[StepInfo], None]
    every: int


class TrainerHooks:
    hooks: List[_Hook] = []

    def run_hooks(self, info: StepInfo, force: bool = False):
        for hook in self.hooks:
            if force or info.step % hook.every == 0:
                hook.fn(info)

    def add_hook(self, fn: Optional[Callable[[StepInfo], Any]] = None, *, every: int = 1):
        def decorator(fn: Callable[[StepInfo], None]):
            self.hooks.append(_Hook(fn, every))

        if fn is None:
            return decorator
        else:
            return decorator(fn)
