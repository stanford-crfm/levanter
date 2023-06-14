import abc
from typing import Generic, Optional, TypeVar

import equinox as eqx
from jax.random import PRNGKey

from haliax import Axis, NamedArray


MConfig = TypeVar("MConfig")


class LmHeadModel(eqx.Module, Generic[MConfig], abc.ABC):
    """
    Superclass for models with a language modeling head.
    """

    config: MConfig = eqx.static_field()

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: MConfig, *, key: PRNGKey) -> "LmHeadModel[MConfig]":
        pass

    @abc.abstractmethod
    def __call__(self, input_ids: NamedArray, attn_mask: Optional[NamedArray] = None, *, inference, key) -> NamedArray:
        pass
