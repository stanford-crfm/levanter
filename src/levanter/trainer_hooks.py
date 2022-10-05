from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

import equinox as eqx
from chex import PRNGKey


S = TypeVar("S")  # State
B = TypeVar("B", covariant=True)  # Batch
Aux = TypeVar("Aux", covariant=True)  # Auxiliary per-iteration results
T = TypeVar("T")


@dataclass
class StepInfo:
    step: int
    model: eqx.Module
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


#
# def engine_from_loss_fn(
#         model: T, train_loader,
#         loss_and_grad_fn: Callable[[T, B, PRNGKey], Any],
#         optimizer: optax.GradientTransformation,
#         key: PRNGKey)-> Tuple[Engine, ModelAndOptState]:
#     def train_step(state, batch, key):
#         model, opt_state = state
#         loss, grads = loss_and_grad_fn(model, batch, key)
#         updates, opt_state = optimizer.update(grads, opt_state)
#         model = eqx.apply_updates(model, updates)
#         return (model, opt_state), loss
#
#     return Engine(train_step, train_loader, key), ModelAndOptState(model, optimizer.init(model))
