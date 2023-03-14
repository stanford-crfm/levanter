from typing import Callable, Optional

import jax.numpy as jnp

import haliax as hax
from haliax import NamedArray
from haliax.nn import cross_entropy_loss


def next_token_loss(
    SeqLen: hax.Axis,
    Vocab: hax.Axis,
    true_ids: NamedArray,
    pred_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    loss_fn: Callable[[NamedArray, hax.Axis, NamedArray], NamedArray] = cross_entropy_loss,
):
    # need to roll the target tokens back by one so that each token is predicting the next token
    target_y = hax.roll(true_ids, -1, SeqLen)
    target_y = hax.nn.one_hot(target_y, Vocab, dtype=pred_ids.dtype)

    not_last_loss_mask = 1 - hax.nn.one_hot(-1, SeqLen, dtype=jnp.float32)  # one everywhere except the last token
    if loss_mask is not None:
        loss_mask = loss_mask * not_last_loss_mask
    else:
        loss_mask = not_last_loss_mask

    loss = loss_fn(pred_ids, Vocab, target_y)
    loss = hax.mean(loss, where=loss_mask, axis=SeqLen)

    return loss
