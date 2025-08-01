import abc
import dataclasses
from typing import Generic, TypeVar

import draccus
import equinox as eqx
import jax.numpy as jnp
import optax


S = TypeVar("S")
M = TypeVar("M")


class ModelAveraging(eqx.Module, Generic[M]):
    """
    This is the interface for model averaging algorithms. Model averaging algorithms are used to average
    the parameters of a model over multiple training steps. This is useful for improving generalization
    """

    @abc.abstractmethod
    def update(self: S, model: M, step: int) -> S:
        pass

    @property
    @abc.abstractmethod
    def model_params(self) -> M:
        pass


class EmaModelAveraging(ModelAveraging[M]):
    """
    Exponential moving average model averaging
    """

    model: M
    beta: float = eqx.field(static=True)

    def update(self: S, new_model: M, step: int) -> S:
        del step
        # 1 - beta because increment_update expects the weight of the new model
        return dataclasses.replace(self, model=optax.incremental_update(new_model, self.model, 1 - self.beta))  # type: ignore

    @property
    def model_params(self) -> M:
        return self.model


class EmaDecaySqrtModelAveraging(ModelAveraging[M]):
    """Hybrid EMA followed by :math:`1 - \sqrt{x}` decay."""

    model: M
    switch_step: int = eqx.field(static=True)
    decay_steps: int = eqx.field(static=True)
    beta: float = eqx.field(static=True, default=0.999)
    epsilon: float = eqx.field(static=True, default=1e-5)

    def update(self, new_model: M, step: int) -> "EmaDecaySqrtModelAveraging[M]":
        if step < self.switch_step:
            alpha = 1.0 - self.beta
        else:
            t = step - self.switch_step
            frac = jnp.clip(t / self.decay_steps, 0.0, 1.0)
            alpha = 1.0 - jnp.sqrt(frac)

        alpha = jnp.maximum(alpha, self.epsilon)
        updated = optax.incremental_update(new_model, self.model, alpha)
        return dataclasses.replace(self, model=updated)  # type: ignore[arg-type]

    @property
    def model_params(self) -> M:
        return self.model


class ModelAveragingConfig(abc.ABC, draccus.ChoiceRegistry, Generic[M]):
    @abc.abstractmethod
    def create(self, model: M) -> ModelAveraging[M]:
        pass


@ModelAveragingConfig.register_subclass("ema")
@dataclasses.dataclass(frozen=True)
class EmaModelAveragingConfig(ModelAveragingConfig[M]):
    beta: float = 0.999

    def create(self, model: M) -> EmaModelAveraging[M]:
        return EmaModelAveraging(model=model, beta=self.beta)


@ModelAveragingConfig.register_subclass("ema_decay_sqrt")
@dataclasses.dataclass(frozen=True)
class EmaDecaySqrtConfig(ModelAveragingConfig[M]):
    """EMA followed by :math:`1 - \sqrt{x}` decay."""

    beta: float = 0.999
    switch_step: int = 100_000
    decay_steps: int = 100_000
    epsilon: float = 1e-5

    def create(self, model: M) -> EmaDecaySqrtModelAveraging[M]:
        return EmaDecaySqrtModelAveraging(
            model=model,
            beta=self.beta,
            switch_step=self.switch_step,
            decay_steps=self.decay_steps,
            epsilon=self.epsilon,
        )
