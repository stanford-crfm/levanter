from dataclasses import dataclass
from typing import Callable, TypeVar, Generic, Tuple, Iterable, List, Any

import jax.random as jrandom
import equinox as eqx
import optax
from chex import PRNGKey

S = TypeVar('S')  # State
B = TypeVar('B', covariant=True)  # Batch
Aux = TypeVar('Aux', covariant=True)  # Auxiliary per-iteration results
T = TypeVar('T')

@dataclass
class StepInfo(Generic[S, B, Aux]):
    step: int
    state: S
    batch: B
    aux: Aux
    next_key: jrandom.PRNGKey


@dataclass
class ModelAndOptState:
    model: eqx.Module
    opt_state: Any



class Engine(Generic[S, B, Aux]):
    hooks: List[Callable[[StepInfo[S, B, Aux]], None]] = []
    """
    Loosely inspired by Ignite's engine, mostly just provides a convenient way to attach hooks to a training loop.
    """
    # TODO: do we want to commit to the key being managed by the engine?
    def __init__(self, train_step_fn: Callable[[S, B, PRNGKey], Tuple[S, Aux]], train_loader: Iterable[B], rng_key: jrandom.PRNGKey):
        self.train_step_fn = train_step_fn
        self.train_loader = train_loader
        self.rng_key = rng_key

    def steps(self, initial_state: S) -> Iterable[StepInfo[S, B, Aux]]:
        state = initial_state
        key = self.rng_key
        for (i, batch) in enumerate(self.train_loader):
            key, split_key = jrandom.split(key, 2)
            state, aux = self.train_step_fn(state, batch, key)
            split_key = key
            yield StepInfo(i, state, batch, aux, split_key)

    def on_iter(self, every: int = 1):
        def decorator(fn: Callable[[StepInfo[S, B, Aux]], None]):
            if every == 1:
                self.hooks.append(fn)
            else:
                self.hooks.append(lambda info: fn(info) if info.step % every == 0 else None)
            return fn
        return decorator

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
