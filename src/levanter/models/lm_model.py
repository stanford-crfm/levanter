import abc
from typing import Generic, Optional, Type, TypeVar

import draccus
from jax.random import PRNGKey

from haliax import Axis, NamedArray


LmConfigT = TypeVar("LmConfigT", bound="LmConfig")
LmT = TypeVar("LmT", bound="LmHeadModel")


# TODO: for some reason, mypy doesn't like the discover_packages_path argument?
class LmConfig(draccus.PluginRegistry, abc.ABC, Generic[LmT], discover_packages_path="levanter.models"):  # type: ignore
    @property
    @abc.abstractmethod
    def model_type(cls) -> Type[LmT]:
        pass

    @property
    @abc.abstractmethod
    def KeyPos(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Pos(self) -> Axis:
        pass

    def build(self, Vocab: Axis, *, key: PRNGKey) -> "LmT":
        return self.model_type.init(Vocab, self, key=key)  # type: ignore


class LmHeadModel(Generic[LmConfigT], abc.ABC):
    """
    Superclass for models with a language modeling head.
    """

    @property
    @abc.abstractmethod
    def config(self) -> LmConfigT:
        pass

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: LmConfigT, *, key: PRNGKey) -> "LmHeadModel[LmConfigT]":
        pass

    @abc.abstractmethod
    def __call__(
        self, input_ids: NamedArray, attn_mask: Optional[NamedArray] = None, *, inference, key=None
    ) -> NamedArray:
        pass
