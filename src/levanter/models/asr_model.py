import abc
from typing import Optional

import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn import cross_entropy_loss

from levanter.data.audio import AudioTextExample
from levanter.models.attention import AttentionMask


class ASRMixin(abc.ABC):
    """
    Superclass for models performing ASR
    """

    @property
    @abc.abstractmethod
    def Vocab(self) -> Axis:
        pass

    @property
    @abc.abstractmethod
    def Pos(self) -> Axis:
        pass

    @abc.abstractmethod
    def resize_vocab(self, new_size: int, key: Optional[PRNGKey] = None) -> "ASRMixin":
        """
        Resizes the vocabulary of the ASR Output space. Key may be provided to use random initialization, otherwise,
        there should be some deterministic initialization of any new parameters.
        """
        pass

    @abc.abstractmethod
    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        pass

    def compute_loss(
        self,
        example: AudioTextExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> jnp.ndarray | NamedArray:
        """
        Computes the cross-entropy loss for predicted ASR tokens. If reduction is not None, the loss is reduced
        across the reduction axis (with reduction_axis=None meaning all axes). If reduction is None, the loss is not
        reduced, and the result is a named array with axes (*batch axes, sequence_length).
        """
        logits = self(example.audio, example.tokens, example.attn_mask, key=key)
        logits = logits.astype(jnp.float32)
        targets = hax.roll(example.tokens, -1, axis=self.Pos.name)
        target_y = hax.nn.one_hot(targets, self.Vocab, dtype=logits.dtype)
        loss = cross_entropy_loss(
            logits, self.Vocab, target_y, reduction, reduction_axis=reduction_axis, where=example.loss_mask
        )

        return loss

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size
