import abc
from typing import Generic, Optional, Type, TypeVar

import draccus
import equinox as eqx
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn import cross_entropy_loss

from levanter.models.attention import AttentionMask


LmConfigT = TypeVar("LmConfigT", bound="LmConfig")
LmT = TypeVar("LmT", bound="LmHeadModel")


class LmExample(eqx.Module):
    tokens: hax.NamedArray
    loss_mask: hax.NamedArray
    attn_mask: AttentionMask | NamedArray = AttentionMask.causal()

    @staticmethod
    def causal(
        tokens: hax.NamedArray, *, loss_mask: Optional[hax.NamedArray] = None, ignore_id: Optional[int] = None
    ) -> "LmExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        Pos = tokens.axes[0]

        # don't predict the last token.
        if loss_mask is None:
            loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)

        if ignore_id is not None:
            # we don't compute loss for any tokens matching the ignore index
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_mask = loss_mask * ignore_mask

        attn_mask = AttentionMask.causal()
        return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)


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

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        return None

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

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @property
    def KeyPos(self) -> Axis:
        return self.config.KeyPos

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: LmConfigT, *, key: PRNGKey) -> "LmHeadModel[LmConfigT]":
        pass

    @abc.abstractmethod
    def __call__(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        pass

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKey] = None) -> "LmHeadModel[LmConfigT]":
        """
        Resizes the vocabulary of the model. Key may be provided to use random initialization, otherwise, there
        should be some deterministic initialization of any new parameters.
        """
        pass

    def compute_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> jnp.ndarray | NamedArray:
        """
        Computes the cross-entropy loss for a language modeling example. If reduction is not None, the loss is reduced
        across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
        reduced, and the result is a named array with axes (*batch axes, sequence_length).
        """
        logits = self(example.tokens, example.attn_mask, key=key)
        extras = None
        if isinstance(logits, tuple):
            assert len(logits) == 2
            logits, extras = logits
        # TODO: would be nice if we made the dtype configurable
        logits = logits.astype(jnp.float32)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        loss = cross_entropy_loss(
            logits, self.Vocab, target_y, reduction, reduction_axis=reduction_axis, where=example.loss_mask
        )

        return loss, extras

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size
