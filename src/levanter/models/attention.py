from typing import Optional, overload

import equinox as eqx

import haliax
from haliax import Axis, NamedArray
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or


class AttentionMask(eqx.Module):
    """

    !!! warning
        This class is still experimental. I'm not super happy with it yet.

    Represents an attention mask in a structured way to make it easier to optimize attention for particular use cases
    (causal, prefix, etc.). It is anticipated that this will be extended with new types of masks as needed.

    In general, it should be safe to batch Attention Masks, but it is important that *all members of a batch have the
    same sequence of combined masks*. Otherwise, the batching will not work and you'll get weird errors

    The interface exposed by this class is designed to work well with the attention functions in this module as
    well as something like flash attention.

    A mask can be materialized, in which case it returns the mask as a NamedArray.
    """

    is_causal: bool = eqx.static_field()
    explicit_mask: Optional[NamedArray] = None
    # TODO: add sequence packing
    # TODO: add prefixlm
    # cf https://github.com/google-research/t5x/blob/51a99bff8696c373cc03918707ada1e98cbca407/t5x/examples/decoder_only/layers.py#L978

    def materialize(
        self, QPos: Axis, KPos: Axis, q_slice: Optional[haliax.dslice] = None, k_slice: Optional[haliax.dslice] = None
    ) -> Optional[NamedArray]:
        """
        Materialize the mask as a NamedArray. This is useful for attention functions that don't support masks,
        or for the inner loop
        """
        if q_slice is None:
            q_slice = haliax.dslice(0, QPos.size)
        if k_slice is None:
            k_slice = haliax.dslice(0, KPos.size)

        if self.is_causal:
            causal = causal_mask(QPos.resize(q_slice.size), KPos.resize(k_slice.size), q_slice.start, k_slice.start)
        else:
            causal = None

        if self.explicit_mask is not None:
            explicit = self.explicit_mask[QPos.name, q_slice, KPos.name, k_slice]
        else:
            explicit = None

        return combine_masks_and(causal, explicit)

    @staticmethod
    def causal() -> "AttentionMask":
        return AttentionMask(is_causal=True)

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(explicit_mask=mask)

    def __and__(self, other) -> "AttentionMask":
        is_causal = self.is_causal and other.is_causal
        explicit_mask = combine_masks_and(self.explicit_mask, other.explicit_mask)
        return AttentionMask(is_causal=is_causal, explicit_mask=explicit_mask)

    def __or__(self, other) -> "AttentionMask":
        is_causal = self.is_causal or other.is_causal
        explicit_mask = combine_masks_or(self.explicit_mask, other.explicit_mask)
        return AttentionMask(is_causal=is_causal, explicit_mask=explicit_mask)


@overload
def materialize_mask(
    mask: NamedArray | AttentionMask,
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> NamedArray:
    ...


@overload
def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    ...


def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    """
    Materialize an attention mask if it is an AttentionMask. Otherwise, just return it.
    """
    if isinstance(mask, AttentionMask):
        mask = mask.materialize(QPos, KPos, q_slice=q_slice, k_slice=k_slice)
        return mask
    elif isinstance(mask, NamedArray):
        if q_slice is not None or k_slice is not None:
            if q_slice is None:
                q_slice = haliax.dslice(0, QPos.size)
            if k_slice is None:
                k_slice = haliax.dslice(0, KPos.size)
            mask = mask[QPos.name, q_slice, KPos.name, k_slice]

        return mask
    else:
        assert mask is None
        return None


# TODO: padding mask
# TODO: FCM mask?
# TODO: sequence packing mask
