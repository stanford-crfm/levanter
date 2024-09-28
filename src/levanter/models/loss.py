from typing import Optional

import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray
from haliax.nn import cross_entropy_loss_and_log_normalizers


def next_token_loss(
    Pos: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_ids: NamedArray,
    true_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
):
    Pos, Vocab = pred_ids.resolve_axis((Pos, Vocab))
    # need to roll the target tokens back by one so that each token is predicting the next token
    target_y = hax.roll(true_ids, -1, Pos)
    target_y = hax.nn.one_hot(target_y, Vocab, dtype=pred_ids.dtype)  # type: ignore

    # one everywhere except the last token
    not_last_loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)  # type: ignore
    if loss_mask is not None:
        loss_mask = loss_mask * not_last_loss_mask
    else:
        loss_mask = not_last_loss_mask

    return cross_entropy_and_logsumexp_penalty(
        pred_ids,
        Vocab,
        target_y,
        reduction=reduction,
        reduction_axis=reduction_axis,
        where=loss_mask,
        logsumexp_weight=logsumexp_weight,
    )


def cross_entropy_and_logsumexp_penalty(
    pred_y: NamedArray,
    Vocab: hax.Axis,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    where: Optional[NamedArray] = None,
    logsumexp_weight=0.0,
) -> NamedArray:
    """A loss function that combines cross entropy loss with a logsumexp penalty."""

    loss, log_normalizers = cross_entropy_loss_and_log_normalizers(pred_y, Vocab, target_y)

    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where)
