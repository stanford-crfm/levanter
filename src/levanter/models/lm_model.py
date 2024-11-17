import abc
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar

import draccus
import equinox as eqx
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray

from levanter.models.attention import AttentionMask
from levanter.models.loss import next_token_loss


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
@dataclass(frozen=True)
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

    @property
    @abc.abstractmethod
    def Embed(self) -> Axis:
        pass

    cross_entropy_block_size: Optional[int] = None
    """
    The block size for computing cross-entropy loss. This is the number of tokens that are processed together
    in a single block. This can be adjusted to fit within memory constraints. It's deliberately set to a large
    value because it usually faster to compute the loss in larger blocks.
    """

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        return None

    def build(self, Vocab: Axis, *, key: PRNGKey) -> "LmT":
        return self.model_type.init(Vocab, self, key=key)  # type: ignore


class LmHeadModel(eqx.Module, Generic[LmConfigT]):
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

    @property
    def Embed(self) -> Axis:
        return self.config.Embed

    @classmethod
    @abc.abstractmethod
    def init(cls, Vocab: Axis, config: LmConfigT, *, key: PRNGKey) -> "LmHeadModel[LmConfigT]":
        pass

    def __call__(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        """
        Compute the logits for the next token in a sequence.
        Args:
            input_ids: token IDs with shape [..., Pos]
            attn_mask: attention mask with shape [..., Pos, KeyPos]
            key: PRNGKey for random number generation

        Returns:
            NamedArray: logits with shape [..., Pos, Vocab]

        """
        x = self.activations(input_ids, attn_mask, key=key)
        lm_logits = hax.dot(x, self.get_lm_head(), axis=self.Embed)

        return lm_logits

    @abc.abstractmethod
    def activations(
        self, input_ids: NamedArray, attn_mask: Optional[AttentionMask | NamedArray] = None, *, key=None
    ) -> NamedArray:
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKey for random number generation

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        pass

    @abc.abstractmethod
    def get_lm_head(self) -> hax.NamedArray:
        """
        The language modeling head of the model. Should have shape {Embed, Vocab}.
        """
        raise NotImplementedError("get_lm_head not implemented")

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKey] = None) -> "LmHeadModel[LmConfigT]":
        """
        Resizes the vocabulary of the model. Key may be provided to use random initialization, otherwise, there
        should be some deterministic initialization of any new parameters.
        """
        pass

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size


def compute_next_token_loss(
    model: LmHeadModel,
    example: LmExample,
    *,
    key=None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    loss_dtype: Optional[Type[jnp.dtype]] = jnp.float32,
) -> jnp.ndarray | NamedArray:
    """
    Computes the cross-entropy loss for a language modeling example. If reduction is not None, the loss is reduced
    across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
    reduced, and the result is a named array with axes (*batch axes, sequence_length).
    """
    activations = model.activations(example.tokens, example.attn_mask, key=key)

    loss = next_token_loss(
        model.Pos,
        model.Embed,
        model.Vocab,
        activations,
        model.get_lm_head(),
        example.tokens,
        loss_mask=example.loss_mask,
        reduction=reduction,
        reduction_axis=reduction_axis,
        logsumexp_weight=logsumexp_weight,
        dtype=loss_dtype,
        block_size=model.config.cross_entropy_block_size,
    )

    return loss
