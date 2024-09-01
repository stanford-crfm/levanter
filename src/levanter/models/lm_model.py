import abc
from typing import Generic, Optional, Type, TypeVar

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn import cross_entropy_loss

from levanter.models.attention import AttentionMask

LmConfigT = TypeVar("LmConfigT", bound="LmConfig")
LmT = TypeVar("LmT", bound="LmHeadModel")

class MaskedLmExample(eqx.Module):
    tokens: hax.NamedArray
    loss_mask: hax.NamedArray
    attn_mask: hax.NamedArray
    targets: Optional[hax.NamedArray] = None

    @staticmethod
    def masked_lm(
        tokens: hax.NamedArray, targets: hax.NamedArray, attn_mask: hax.NamedArray, mask_token_id: Optional[int] = None
    ) -> "MaskedLmExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        if tokens.shape != targets.shape:
            raise ValueError("tokens and targets must have the same shape")

        Pos = tokens.axes[0]

        mask = tokens.array != targets.array
        loss_mask = hax.named(mask.astype(jnp.float32), Pos)

        if mask_token_id is not None:
            ignore_mask = targets.array != mask_token_id
            loss_mask = loss_mask * hax.named(ignore_mask.astype(jnp.float32), Pos)

        return MaskedLmExample(tokens=tokens, targets=targets, loss_mask=loss_mask, attn_mask=attn_mask)


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

        if loss_mask is None:
            loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)

        if ignore_id is not None:
            ignore_mask = hax.roll(tokens, -1, Pos) != ignore_id
            loss_mask = loss_mask * ignore_mask

        attn_mask = AttentionMask.causal()
        return LmExample(tokens=tokens, loss_mask=loss_mask, attn_mask=attn_mask)


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
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attn_mask shape: {attn_mask.shape}")

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKey] = None) -> "LmHeadModel[LmConfigT]":
        pass

    def compute_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> jnp.ndarray | NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key)
        logits = logits.astype(jnp.float32)
        if example.targets is not None:
            targets = example.targets
        else:
            targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        loss = cross_entropy_loss(
            logits, self.Vocab, target_y, reduction, reduction_axis=reduction_axis, where=example.loss_mask
        )

        return loss

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size