import dataclasses
import typing
from typing import Generic, TypeVar

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import PRNGKeyArray, PyTree
from optax import OptState

from haliax.types import IntScalar
from levanter.types import FilterSpec
from levanter.utils.jax_utils import is_inexact_arrayish

M = TypeVar("M", bound=PyTree)

class TrainerStateLike(typing.Protocol[M]):

S = TypeVar("S", bound=eqx.Module)


def _ensure_int_is_array(x):
    # who tf decided that bools are ints
    if isinstance(x, int) and not isinstance(x, bool):
        return jnp.array(x)
    else:
        return x


class TrainerState(eqx.Module, Generic[M]):
    """
    This is the state of the trainer. It contains the model, optimizer state, and random key.
    It is an equinox Module because it is a PyTree that gets passed to the core `train_step` method
    of the Trainer. This unfortunately means that `step` is an Array and not an int, hence the IntScalar.

    It's designed to be extended by subclasses. Alternatively, you can implement your own trainer state
    that doesn't inherit from this class.
    """

    step: IntScalar = eqx.field(converter=_ensure_int_is_array)
    model: M
    opt_state: OptState
    training_key: PRNGKeyArray
    is_trainable: PyTree[FilterSpec]  # = eqx.field(static=True)

    @property
    def int_step(self) -> int:
        """
        Returns the step as an int. On multinode, doing
        """
        return int(self.step)

    @property
    def trainable_model(self) -> M:
        return eqx.filter(self.model, self.is_trainable)


def init_optimizer_for_trainables(optimizer, model, is_trainable):
    trainable = trainables_only(model, is_trainable)
    opt_state = optimizer.init(trainable)
    return opt_state


def _params_only(t):
    return eqx.filter(t, is_inexact_arrayish)


def _partition_trainable_params(model, filter):
    """
    Partitions the model into trainable and non-trainable parameters. This is used internally
    for the gradient calculation and checkpointing, but you can also use it to filter out params for logging
    or something.

    Returns:
        trainable, non-trainable
    """

    def trainable_and_diffable(pred):
        if callable(pred):
            return lambda x: pred(x) and is_inexact_arrayish(x)
        elif pred is True:
            return is_inexact_arrayish
        else:
            return pred

    combined_mask = jax.tree_util.tree_map(trainable_and_diffable, filter)
    return eqx.partition(model, combined_mask)


def trainables_only(model, filter):
    """
    Filters out non-trainable parameters from the model. This is used internally to
    for the optimizer state and to compute gradients, but you can also use it to filter out
    params for logging or something.
    """
    return _partition_trainable_params(model, filter)[0]


def cast_params_by_trainability(model, mp, is_trainable):
    """
    Casts the parameters of a model to the appropriate precision based on the is_trainable filter spec.
    Trainable parameters are cast to param precision, non-trainable parameters are cast to compute precision.
    """

    trainable, non_trainable = _partition_trainable_params(model, is_trainable)
    trainable = mp.cast_to_param(trainable)
    non_trainable = mp.cast_to_compute(non_trainable)
    model = eqx.combine(trainable, non_trainable)
    return model


def saveable_training_mask(trainer_state: S | typing.Type[S], is_trainable_param: FilterSpec = True) -> FilterSpec:
    """
    Returns a mask representing the saveable portion of a trainer state. This is used to filter out non-trainable
    parameters for checkpointing and for logging.

    This method works with both instances of a trainer state and the type of a trainer state. If you pass in a type,
    it must be a dataclass that won't validate its constructor arguments (or at least not throw an error if you pass
    in all True for all fields).
    """
    # when we have a Type, we want to instantiate it with all True for all args (regardless of the actual types)
    # have to do dataclass magic to instantiate the class with all True
    if isinstance(trainer_state, type):
        fields = dataclasses.fields(trainer_state)
        trainer_state = trainer_state(*[True] * len(fields))
    else:
        trainer_state = jax.tree_util.tree_map(lambda x: True, trainer_state)
    saveable_state = dataclasses.replace(trainer_state, model=is_trainable_param)  # type: ignore
    return saveable_state
