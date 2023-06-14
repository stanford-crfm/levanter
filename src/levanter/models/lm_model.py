import abc
from typing import Generic, Optional, Type, TypeVar

from jax.random import PRNGKey

from haliax import Axis, NamedArray


LmConfigT = TypeVar("LmConfigT", bound="LmConfig")
LmT = TypeVar("LmT", bound="LmHeadModel")


class LmConfig(abc.ABC, Generic[LmT]):
    @property
    @abc.abstractmethod
    def model_type(self) -> Type[LmT]:
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
