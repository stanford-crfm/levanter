import abc
import dataclasses
import functools as ft
from typing import Optional, Tuple, TypeAlias, Union, overload

import equinox as eqx

import haliax
from haliax import Axis, AxisSelector, NamedArray
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or, prefix_lm_mask


AttnMask: TypeAlias = Union[NamedArray, "AttentionMask"]


@overload
def materialize_mask(mask: AttnMask) -> NamedArray:
    ...


@overload
def materialize_mask(mask: Optional[AttnMask]) -> Optional[NamedArray]:
    ...


def materialize_mask(mask: Optional[AttnMask]) -> Optional[NamedArray]:
    """
    Materialize an attention mask if it is an AttentionMask. Otherwise, just return it.
    """
    if isinstance(mask, AttentionMask):
        mask = mask.materialize()
    return mask


class AttentionMask(eqx.Module, abc.ABC):
    """
    Represents an attention mask in a structured way to make it easier to optimize attention for particular use cases
    (causal, prefix, etc.). It is anticipated that this will be extended with new types of masks as needed.

    In general, it should be safe to batch Attention Masks, but it is important that *all members of a batch have the
    same sequence of combined masks*. Otherwise, the batching will not work and you'll get weird errors

    The interface exposed by this class is designed to work well with the attention functions in this module as
    well as something like flash attention.

    A mask can be materialized, in which case it returns the mask as a NamedArray.
    We can also ask for slices of a mask along a particular axis, or for a blocked version of the mask.

    The blocked version of the mask is basically a projection of the mask onto a smaller mask, where each position
    in the smaller mask is the max of the corresponding positions in the larger mask. This is useful for
    blockwise attention mechanisms, like flash or longformer.
    """

    @abc.abstractmethod
    def materialize(self) -> Optional[NamedArray]:
        """
        Materialize the mask as a NamedArray. This is useful for attention functions that don't support masks,
        or for the inner loop
        """
        raise NotImplementedError

    @abc.abstractmethod
    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        """
        Slice the mask along a particular axis. This is useful for extracting a particular slice of a mask
        for use in blocked attention.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        """
        Return a blocked version of the mask. This is useful for blockwise attention mechanisms, like flash or longformer.
        :param axis:
        :param block_size:
        :return:
        """
        raise NotImplementedError

    def __and__(self, other) -> "AttentionMask":
        if isinstance(self, AndAttentionMask):
            conjuncts = list(self.conjuncts)
        else:
            conjuncts = [self]

        if isinstance(other, AndAttentionMask):
            conjuncts.extend(other.conjuncts)
        else:
            conjuncts.append(other)

        return AndAttentionMask(conjuncts)

    def __or__(self, other) -> "AttentionMask":
        if isinstance(self, OrAttentionMask):
            disjuncts = list(self.disjuncts)
        else:
            disjuncts = [self]

        if isinstance(other, OrAttentionMask):
            disjuncts.extend(other.disjuncts)
        else:
            disjuncts.append(other)

        return OrAttentionMask(disjuncts)


class CausalMask(AttentionMask):
    Pos: Axis = eqx.field(static=True)
    KeyPos: Axis = eqx.field(static=True)
    pos_start: int = eqx.field(static=True, default=0)
    kpos_start: int = eqx.field(static=True, default=0)

    def materialize(self) -> Optional[NamedArray]:
        return causal_mask(self.Pos, self.KeyPos, self.pos_start, self.kpos_start)

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        if haliax.selects_axis(axis, self.Pos):
            return dataclasses.replace(self, Pos=self.Pos.resize(length), pos_start=self.pos_start + start)
        elif haliax.selects_axis(axis, self.KeyPos):
            return dataclasses.replace(self, KeyPos=self.KeyPos.resize(length), kpos_start=self.kpos_start + start)
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")

    def blocked(self, axis: AxisSelector, block_size: int) -> "CausalMask":
        # a blocked causal mask is just a smaller causal mask
        if haliax.selects_axis(axis, self.Pos):
            if self.Pos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.Pos.size} with block size {block_size}")
            new_size = self.Pos.size // block_size
            return dataclasses.replace(self, Pos=self.Pos.resize(new_size), pos_start=self.pos_start // block_size)
        elif haliax.selects_axis(axis, self.KeyPos):
            if self.KeyPos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.KeyPos.size} with block size {block_size}")
            new_size = self.KeyPos.size // block_size
            return dataclasses.replace(
                self, KeyPos=self.KeyPos.resize(new_size), kpos_start=self.kpos_start // block_size
            )
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")


class PrefixAttentionMask(AttentionMask):
    Pos: Axis = eqx.field(static=True)
    KeyPos: Axis = eqx.field(static=True)
    # TODO: prefix size needs to be dynamic
    prefix_size: int = eqx.field(static=True)
    pos_start: int = eqx.field(static=True, default=0)
    kpos_start: int = eqx.field(static=True, default=0)

    def materialize(self) -> Optional[NamedArray]:
        return prefix_lm_mask(self.Pos, self.KeyPos, self.prefix_size, self.pos_start, self.kpos_start)

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        if haliax.selects_axis(axis, self.Pos):
            return dataclasses.replace(self, Pos=self.Pos.resize(length), pos_start=self.pos_start + start)
        elif haliax.selects_axis(axis, self.KeyPos):
            return dataclasses.replace(self, KeyPos=self.KeyPos.resize(length), kpos_start=self.kpos_start + start)
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        if haliax.selects_axis(axis, self.Pos):
            if self.Pos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.Pos.size} with block size {block_size}")
            new_size = self.Pos.size // block_size
            return dataclasses.replace(self, Pos=self.Pos.resize(new_size), pos_start=self.pos_start // block_size)
        elif haliax.selects_axis(axis, self.KeyPos):
            if self.KeyPos.size % block_size != 0:
                raise ValueError(f"Cannot block mask of size {self.KeyPos.size} with block size {block_size}")
            new_size = self.KeyPos.size // block_size
            return dataclasses.replace(
                self, KeyPos=self.KeyPos.resize(new_size), kpos_start=self.kpos_start // block_size
            )
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.Pos} and {self.KeyPos}")


class ExplicitMask(AttentionMask):
    mask: NamedArray

    def materialize(self) -> Optional[NamedArray]:
        return self.mask

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        if haliax.selects_axis(self.mask.axes, axis):
            return dataclasses.replace(self, mask=self.mask.slice(axis, start=start, length=length))
        else:
            raise ValueError(f"Invalid axis {axis}. Valid axes are {self.mask}")

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        # we have to do blocked ourselves, and it's a bit messy
        axis = self.mask.resolve_axis(axis)

        if axis.size % block_size != 0:
            raise ValueError(f"Cannot block mask axis of size {axis.size} with block size {block_size}")

        new_size = self.mask.size // block_size

        block_axis = axis.alias(axis.name + "__block").resize(block_size)
        unflattened = self.mask.unflatten_axis(axis, (axis.resize(new_size), block_axis))
        blocked = haliax.any(unflattened, axis=block_axis)

        return dataclasses.replace(self, mask=blocked)


class AndAttentionMask(AttentionMask):
    conjuncts: Tuple[AttentionMask, ...]

    def materialize(self) -> Optional[NamedArray]:
        return ft.reduce(combine_masks_and, (conj.materialize() for conj in self.conjuncts))

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        return dataclasses.replace(self, conjuncts=tuple(conj.slice(axis, start, length) for conj in self.conjuncts))

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        return dataclasses.replace(self, conjuncts=tuple(conj.blocked(axis, block_size) for conj in self.conjuncts))


class OrAttentionMask(AttentionMask):
    disjuncts: Tuple[AttentionMask, ...]

    def materialize(self) -> Optional[NamedArray]:
        return ft.reduce(combine_masks_or, (disj.materialize() for disj in self.disjuncts))

    def slice(self, axis: AxisSelector, start: int, length: int) -> "AttentionMask":
        return dataclasses.replace(self, disjuncts=tuple(disj.slice(axis, start, length) for disj in self.disjuncts))

    def blocked(self, axis: AxisSelector, block_size: int) -> "AttentionMask":
        return dataclasses.replace(self, disjuncts=tuple(disj.blocked(axis, block_size) for disj in self.disjuncts))


# TODO: padding mask
# TODO: FCM mask?
# TODO: sequence packing mask
