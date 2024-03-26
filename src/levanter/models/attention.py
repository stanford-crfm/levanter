import math
import warnings
from typing import Optional, Union, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax
from haliax import Axis, AxisSelection, AxisSelector, NamedArray
from haliax.jax_utils import named_call
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or
from haliax.types import PrecisionLike


@named_call
def dot_product_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union["AttentionMask", NamedArray]] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    use_flash: bool = False,
    flash_block_size: Optional[int] = None,
    dropout: float = 0.0,
    *,
    inference: bool = True,
    prng: Optional[PRNGKeyArray] = None,
):
    """
    This method is similar to [haliax.nn.attention.dot_product_attention][] but uses the [AttentionMask][] class,
    which we might move to haliax.nn.attention in the future.

    Unlike the Haliax version, it requires that the QPos and KPos already be different.

    Args:
        Key: Size of key dimension
        QPos: Axis of query sequence length. Can be an AxisSpec to attend along more than one axis.
        KPos: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
        query: shape at least {QPos, KeySize}
        key: shape at least {KPos, KeySize}
        value: shape at least {KPos, ValueSize}
        mask: attention mask
        bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
        attention_dtype: Optional dtype to use for attention
        precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
        use_flash: whether to use flash attention
        flash_block_size: block size for flash attention. If None, will use an appropriate default
        dropout: dropout rate
        inference: whether to use inference mode
        prng: PRNGKeyArray for dropout
    Returns:
        NamedArray of shape (value.axes - KPos + QPos)
    """
    if QPos == KPos:
        raise ValueError("QPos and KPos must be different")

    accelerator_type = jax.local_devices()[0].platform

    if not use_flash:
        return simple_attention_with_dropout(
            QPos, KPos, Key, query, key, value, mask, bias, inference, dropout, attention_dtype, precision, prng=prng
        )
    elif accelerator_type == "gpu":
        try:
            return _te_flash_attention(
                QPos,
                KPos,
                Key,
                query,
                key,
                value,
                block_size=flash_block_size,
                mask=mask,
                bias=bias,
                dropout=dropout,
                inference=inference,
                precision=precision,
                prng=prng,
                attention_dtype=attention_dtype,
            )
        except ImportError as e:
            if "transformer_engine" not in str(e):
                raise

            warnings.warn(
                "transformer_engine is not installed. Please install it to use NVIDIA's optimized fused attention. "
                "Falling back to the reference implementation."
            )
        except NotImplementedError as e:
            message = str(e)
            warnings.warn(
                f"Could not use transformer_engine for flash attention: {message}. Falling back to the reference"
            )
        except ValueError as e:
            message = str(e)
            if message.startswith("Unsupported backend="):
                _dtype = attention_dtype or query.dtype
                msg = "TE doesn't work with these arguments. Falling back to the reference implementation.\n"
                "Check nvte_get_fused_attn_backend for supported configurations:\n"
                "https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/fused_attn/fused_attn.cpp#L71"
                if _dtype not in (jnp.float16, jnp.bfloat16, jnp.float8_e5m2, jnp.float8_e4m3fn):
                    msg += f"In particular, TE doesn't support {_dtype} yet."
                warnings.warn(msg)
            else:
                raise

    from levanter.models.flash_attention import flash_attention

    return flash_attention(
        QPos,
        KPos,
        Key,
        query,
        key,
        value,
        block_size=flash_block_size,
        mask=mask,
        bias=bias,
        dropout=dropout,
        inference=inference,
        key=prng,
        dtype=attention_dtype,
        precision=precision,
    )


def simple_attention_with_dropout(
    QPos: Axis,
    KPos: Axis,
    Key: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    inference: bool = False,
    dropout: float = 0.0,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    *,
    prng: Optional[PRNGKeyArray] = None,
):
    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)
    m = materialize_mask(mask, QPos, KPos)
    weights = haliax.nn.attention.dot_product_attention_weights(
        Key, KPos, query, key, mask=m, bias=bias, attention_dtype=attention_dtype, precision=precision
    )
    weights = haliax.nn.dropout(weights, dropout, key=prng, inference=inference)
    return haliax.dot(KPos, weights, value)


def _te_flash_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    dropout: float = 0.0,
    inference: bool = False,
    *,
    prng: Optional[PRNGKeyArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    block_size: Optional[int] = None,
):
    # try:
    #     from transformer_engine.jax.fused_attn import fused_attn_kvpacked
    # except ImportError:
    #     from transformer_engine.jax.fused_attn import cross_fused_attn as fused_attn_kvpacked
    #
    # try:
    #     from transformer_engine.jax.fused_attn import fused_attn_qkvpacked
    # except ImportError:
    #     from transformer_engine.jax.fused_attn import self_fused_attn as fused_attn_qkvpacked
    #
    from transformer_engine.jax.fused_attn import fused_attn  # noqa: F401
    from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType  # noqa: F401

    attention_dtype = attention_dtype or query.dtype
    query = query.astype(attention_dtype)
    key = key.astype(attention_dtype)
    value = value.astype(attention_dtype)

    if precision is not None:
        warnings.warn("precision is not supported for TE fused attention. Ignoring.")

    # references: https://github.com/NVIDIA/TransformerEngine/blob/8255f87f3ee8076db21777795ce15b6ddf8754c0/transformer_engine/jax/fused_attn.py#L31
    # https://github.com/NVIDIA/TransformerEngine/blob/8255f87f3ee8076db21777795ce15b6ddf8754c0/transformer_engine/jax/flax/transformer.py#L269

    q_class, k_class, v_class = _te_bin_and_group_axes_by_function(query, key, value, QPos, KPos, Key)
    q_: jax.Array = _reshape_axes_for_te_bins(query, q_class).array
    k_ = _reshape_axes_for_te_bins(key, k_class).array
    v_ = _reshape_axes_for_te_bins(value, v_class).array

    B, Sq, Hq, D = q_.shape
    Bk, Sk, Hk, Dk = k_.shape

    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)

    # TODO: must Dk == Dv?
    if k_.shape != v_.shape:
        raise ValueError("k and v must have the same axes")

    if B != Bk:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {q_class['B']} != {k_class['B']}")

    if D != Dk:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {q_class['D']} != {k_class['D']}")

    # Mask is generated by transformer engine based on AttnMaskType
    attn_mask_type, fused_attn_mask = _te_materialize_mask(KPos, QPos, B, mask)

    scaling_factor = 1 / math.sqrt(D)
    is_training = not inference

    # TODO: bias type is probably also configurable
    attn_bias_type = AttnBiasType.NO_BIAS
    fused_attn_bias = None
    if bias:
        raise NotImplementedError("Using bias with flash attention on GPU is not currently implemented.")

    # if q_.shape == k_.shape == v_.shape:
    #     # can use self_fused_attn
    #     qkv_ = jnp.stack((q_, k_, v_), axis=2)
    #     attn_output = (
    #         qkv=qkv_,  # jnp.ndarray,
    #         bias=fused_attn_bias,  # jnp.ndarray,
    #         mask=fused_attn_mask,  # jnp.ndarray,
    #         seed=prng,  # jnp.ndarray,
    #         attn_bias_type=attn_bias_type,  # AttnBiasType,
    #         attn_mask_type=attn_mask_type,  # AttnMaskType,
    #         scaling_factor=scaling_factor,  # float,
    #         dropout_probability=dropout,  # float,
    #         is_training=is_training,  # bool,
    #     )
    # elif k_.shape == v_.shape:
    #     kv_ = jnp.stack((k_, v_), axis=2)
    #     attn_output = cross_fused_attn(
    #         q=q_,
    #         kv=kv_,
    #         bias=fused_attn_bias,
    #         mask=fused_attn_mask,
    #         seed=prng,
    #         attn_bias_type=attn_bias_type,
    #         attn_mask_type=attn_mask_type,
    #         scaling_factor=scaling_factor,
    #         dropout_probability=dropout,
    #         is_training=is_training,
    #     )
    # else:
    attn_output = fused_attn(
        q=q_,
        k=k_,
        v=v_,
        bias=fused_attn_bias,
        mask=fused_attn_mask,
        seed=prng,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        scaling_factor=scaling_factor,
        dropout_probability=dropout,
        is_training=is_training,
    )

    # per the TE code, the output is BSHD. we can reshape it to match our axes
    # we have to ungroup the axes, then reshape them to match our expected output
    attn_output = haliax.named(attn_output, ("B", "S", "H", "D"))
    # the output shape is B, S_q, H_q, D_v. Right now we're requiring D_k == D_v
    # we can reshape it to match our expected output
    attn_output = attn_output.unflatten_axis("B", q_class["B"])
    attn_output = attn_output.unflatten_axis("S", q_class["S"])
    attn_output = attn_output.unflatten_axis("H", q_class["H"])
    attn_output = attn_output.unflatten_axis("D", v_class["D"])
    output_axes = eqx.filter_eval_shape(
        simple_attention_with_dropout,
        QPos,
        KPos,
        Key,
        query,
        key,
        value,
        mask,
        bias,
        inference,
        dropout,
        attention_dtype,
        precision,
        prng=prng,
    ).axes
    attn_output = attn_output.rearrange(output_axes)

    return attn_output


def _te_materialize_mask(KPos, QPos, batch_size, mask):
    from transformer_engine.jax.fused_attn import AttnMaskType

    if isinstance(mask, NamedArray):
        raise NotImplementedError(
            "Custom NamedArray masks are not implemented for flash attention. Please pass an AttentionMask object"
        )
    elif isinstance(mask, AttentionMask):
        if mask.causal():
            attn_mask_type = AttnMaskType.CAUSAL_MASK

            fused_attn_mask = mask.materialize(QPos, KPos)

            assert (
                fused_attn_mask is not None
            ), "If AttentionMask is causal, the materialized array should never be None. Something is wrong."

            fused_attn_mask = fused_attn_mask.array
            fused_attn_mask = jnp.dstack([fused_attn_mask] * batch_size)

        else:
            raise NotImplementedError(
                "Non-Causal masks are not implemented for flash attention. Please pass an AttentionMask object"
            )
    else:
        attn_mask_type = AttnMaskType.NO_MASK
        fused_attn_mask = jnp.ones((batch_size, QPos.size, KPos.size))
    return attn_mask_type, fused_attn_mask


_DUMMY_HEAD = "__head__"
_DUMMY_BATCH = "__batch__"


def _te_bin_and_group_axes_by_function(q, k, v, QPos, KPos, Key):
    """
    TransformerEngine's fused attention API is not as expressive as ours is, so we have to do some grouping
    to make this work.

    TE requires Q, K, and V to have shape BSHD (Batch, Sequence, Head, Embed). the size of the axes is a bit flexible,
    with the following conditions:
    - B must be the same for all (TODO: is this true?)
    - S must be the same for K and V. Q's S can be different
    - H: Q's H must be a multiple of K's H (for GQA or MQA)
    - D must be the same for all (TODO: is this true? possibly V can be different)

    We can thus classify the axes in q, k, v by their function and populate the TE axes in the right order
    - Key is D. ATM we're assuming this is a single axis.
    - QPos and KPos are always S
    - the latest other axis that is present in all three is H. If there are no other axes, we'll add a dummy axis
    - Any other axis that is present in all three is B. If there are no other axes, we'll add a dummy axis
    - If there's an axis present in Q and not in K or V, it's an extra H for Q (as part of GQA).
      These go *after* the primary H because GQA wants these to be minor axes
    - If there are any other axes present in one but not all three, it's an error
     (TODO: we could vmap over these?)
    """
    QPos = q.resolve_axis(QPos)
    KPos = k.resolve_axis(KPos)
    Key = q.resolve_axis(Key)

    q_class = {"B": [], "S": [QPos], "H": [], "D": [Key]}
    k_class = {"B": [], "S": [KPos], "H": [], "D": [Key]}
    v_class = {"B": [], "S": [KPos], "H": [], "D": [Key]}

    present_in_all: set[str] = q.shape.keys() & k.shape.keys() & v.shape.keys()
    spoken_for: set[str] = {QPos.name, KPos.name, Key.name}

    # find the primary H axes: which are axes that are:
    # - present in all three
    # - not spoken for already
    # - come after QPos in Q (if there's already a primary H)
    # - not the 0th axis in Q (even if there's no primary H)
    primary_H: list[Axis] = []
    for a in reversed(q.axes[1:]):
        if a.name in present_in_all and a.name not in spoken_for:
            primary_H.append(a)
        elif a == QPos and primary_H:  # better to always have at least one H?
            break  # anything before QPos we'll say is Batch

    # since we added them in reverse order, we need to reverse them
    primary_H.reverse()

    spoken_for.update([ax.name for ax in primary_H])

    # remaining shared axes are batch axes
    batch_axes = [ax for ax in q.axes if ax.name not in spoken_for and ax.name in present_in_all]

    spoken_for.update([ax.name for ax in batch_axes])

    q_class["B"] = batch_axes
    k_class["B"] = batch_axes
    v_class["B"] = batch_axes

    # if there's an axis in q that's not in k or v, it's an extra H for q
    extra_q_H = [ax for ax in q.axes if ax.name not in spoken_for]

    # we want primary_h to be *before* extra_q_H b/c GQA wants these to be minor axes
    q_class["H"] = primary_H + extra_q_H
    k_class["H"] = primary_H
    v_class["H"] = primary_H

    # now we want to detect any non-spoken-for axes. These are errors
    # eventually we can vmapp over these, but for now we'll just raise an error
    for a in k.axes:
        if a.name not in spoken_for:
            raise ValueError(f"Axis {a.name} is present in k but not in q and/or v")

    for a in v.axes:
        if a.name not in spoken_for:
            raise ValueError(f"Axis {a.name} is present in v but not in q and/or k")

    return q_class, k_class, v_class


def _reshape_axes_for_te_bins(q, q_class):
    """
    Reshape the axes of a qkv as BSHD to match the bins in q_class
    """

    def _maybe_flatten(q, axes, name):
        if axes:
            q = q.flatten_axes(axes, name)
        else:
            q = q.broadcast_axis(Axis(name, 1))
        return q

    q = _maybe_flatten(q, q_class["B"], "B")
    q = _maybe_flatten(q, q_class["S"], "S")
    q = _maybe_flatten(q, q_class["H"], "H")
    q = _maybe_flatten(q, q_class["D"], "D")
    q = q.rearrange(("B", "S", "H", "D"))
    return q


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
            explicit = self.explicit_mask[QPos, q_slice, KPos, k_slice]
        else:
            explicit = None

        return combine_masks_and(causal, explicit)

    @staticmethod
    def causal() -> "AttentionMask":
        return AttentionMask(is_causal=True)

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=False, explicit_mask=mask)

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
            mask = mask[QPos, q_slice, KPos, k_slice]

        return mask
    else:
        assert mask is None
        return None


# TODO: padding mask
# TODO: FCM mask?
# TODO: sequence packing mask
