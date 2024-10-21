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



def fused_cross_entropy_and_logsumexp_penalty(
        Vocab: hax.Axis,
        Embed: hax.AxisSelector,
        pred_embeddings: NamedArray,
        pred_lm_head: NamedArray,
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


from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import haliax as hax
from haliax import NamedArray
from haliax.nn import cross_entropy_loss_and_log_normalizers


def next_token_loss(
    Pos: hax.AxisSelector,
    Vocab: hax.AxisSelector,
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    true_ids: NamedArray,
    loss_mask: Optional[NamedArray] = None,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    logsumexp_weight: Optional[float] = None,
    block_size: Optional[int] = None,  # Added block_size parameter
) -> NamedArray:
    """
    Compute the next token loss with optional block-wise processing.

    Args:
        Pos (hax.AxisSelector): Position axis selector.
        Vocab (hax.AxisSelector): Vocabulary axis selector.
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        true_ids (NamedArray): True token IDs.
        loss_mask (Optional[NamedArray]): Mask to apply to the loss.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        logsumexp_weight (Optional[float]): Weight for logsumexp penalty.
        block_size (Optional[int]): Size of each block for processing.

    Returns:
        NamedArray: Computed loss.
    """
    # Resolve axes
    Pos, Vocab = pred_embeddings.resolve_axis((Pos, Vocab))

    # Shift target tokens to predict the next token
    target_y = hax.roll(true_ids, -1, Pos)

    # One-hot encode the target tokens
    target_y = hax.nn.one_hot(target_y, Vocab, dtype=pred_embeddings.dtype)  # type: ignore

    # Create a mask that excludes the last token
    not_last_loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=jnp.float32)  # type: ignore
    if loss_mask is not None:
        loss_mask = loss_mask * not_last_loss_mask
    else:
        loss_mask = not_last_loss_mask

    # Compute the loss with optional block-wise processing
    return fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        pred_lm_head,
        Contract=hax.AxisSelector("embed"),
        Label=Vocab,
        target_y=target_y,
        reduction=reduction,
        reduction_axis=reduction_axis,
        where=loss_mask,
        logsumexp_weight=logsumexp_weight,
        block_size=block_size,  # Pass block_size
    )


def fused_cross_entropy_loss_and_logsumexp_penalty(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.AxisSelector,
    Label: hax.AxisSelector,
    target_y: NamedArray,
    *,
    reduction: Optional[hax.ReductionFunction] = hax.mean,
    reduction_axis: Optional[hax.AxisSelection] = None,
    where: Optional[NamedArray] = None,
    logsumexp_weight: float = 0.0,
    block_size: Optional[int] = None,  # Added block_size parameter
) -> NamedArray:
    """
    Compute the cross-entropy loss and logsumexp penalty using embeddings and lm_head,
    with optional block-wise processing.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        Contract (hax.AxisSelector): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        target_y (NamedArray): One-hot encoded target tokens.
        reduction (Optional[hax.ReductionFunction]): Reduction function.
        reduction_axis (Optional[hax.AxisSelection]): Axis to apply reduction.
        where (Optional[NamedArray]): Mask to apply to the loss.
        logsumexp_weight (float): Weight for logsumexp penalty.
        block_size (Optional[int]): Size of each block for processing.

    Returns:
        NamedArray: Computed loss.
    """
    if block_size is None:
        # Full softmax computation
        logits = hax.dot(pred_embeddings, pred_lm_head, axis=Contract)
        loss, log_normalizers = cross_entropy_loss_and_log_normalizers(logits, Label, target_y)
    else:
        # Block-wise softmax computation
        loss, log_normalizers = block_wise_cross_entropy_loss(pred_embeddings, pred_lm_head, Contract, Label, target_y, block_size)

    if logsumexp_weight != 0.0:
        loss = loss + logsumexp_weight * (log_normalizers**2)

    return hax.nn.loss.maybe_reduce_loss(loss, reduction, reduction_axis, where)


def block_wise_cross_entropy_loss(
    pred_embeddings: NamedArray,
    pred_lm_head: NamedArray,
    Contract: hax.Axis,
    Label: hax.Axis,
    labels_y: NamedArray,
    block_size: int,
) -> Tuple[NamedArray, NamedArray]:
    """
    Compute cross-entropy loss and log normalizers in a block-wise manner without materializing the full logits.

    Args:
        pred_embeddings (NamedArray): Predicted embeddings.
        pred_lm_head (NamedArray): Language model head weights.
        Contract (hax.Axis): Axis to contract over.
        Label (hax.AxisSelector): Label (Vocab) axis.
        labels_y (NamedArray): label tensor.
        block_size (int): Size of each block for processing.

    Notes:
        labels_y being anything other than the label tensor would remove any benefits

        TODO: but if XLA smart enough to optimize it out?

    Returns:
        Tuple[NamedArray, NamedArray]: Tuple of loss and log_normalizers.
    """
    vocab_size = Label.size
    #
    # if num_blocks == 1:
    #     # No need for block-wise processing
    #     logits = hax.dot(pred_embeddings, pred_lm_head, axis=Contract)
    #     labels_y = hax.nn.one_hot(labels_y, Label, dtype=pred_embeddings.dtype)
    #     return cross_entropy_loss_and_log_normalizers(logits, Label, labels_y)
    #
    # ensure block size divides vocab size
    if vocab_size % block_size != 0:
        todo handle stragglers
        # raise ValueError("Vocab size must be a multiple of block size")
        has_stragglers = True
    else:
        has_stragglers = False

    num_blocks = vocab_size // block_size

    # Initialize accumulators: loss, logsumexp, max_logits
    initial_O = hax.zeros(labels_y.axes)
    initial_logsumexp = hax.full(labels_y.axes, -jnp.inf)
    initial_max = hax.full(labels_y.axes, -jnp.inf)
    # We don't need this b/c we're using one-hot targets
    # initial_sumV = hax.full(labels_y.axes, 0.0)

    def process_block(block_idx, acc):
        """
        Process a single block of the Vocab dimension.

        Args:
            block_idx (int): Index of the current block.
            acc (Tuple[NamedArray, NamedArray, jnp.ndarray]): Accumulators for loss, logsumexp, and max logits.

        Returns:
            Tuple[NamedArray, NamedArray, jnp.ndarray]: Updated accumulators
        """
        O_prev, logsumexp_prev, max_prev = acc

        start = block_idx * block_size
        Block = Label.resize(block_size)

        # Materialize the logits for the current block
        lm_head_b = pred_lm_head[Label, hax.dslice(start, block_size)]  # [Contract, Block]
        logits_b = hax.dot(pred_embeddings, lm_head_b, axis=Contract)

        # Update max and logsumexp
        m = hax.maximum(max_prev, hax.max(logits_b, axis=Block))  # [Batch, Seq]
        logsumexp = m + hax.log(hax.exp(logsumexp_prev - m) + hax.sum(hax.exp(logits_b - m), axis=Block))  # [Batch, Seq]

        # Materialize the target for the current block (one-hot)
        target_y_block = _block_one_hot(Block, start, labels_y, logits_b.dtype)  # [Batch, Seq, Block]

        # Update sumV. This is actually unnecessary if we're using one-hot targets
        # sV = sV_prev + hax.sum(target_y_block, axis=Label.name)
        O = O_prev + hax.dot(logits_b, target_y_block, axis=Block)

        return O, logsumexp, m #, sV

    (o, log_z, max_logits) = jax.lax.fori_loop(
        lower=0,
        upper=num_blocks,
        body_fun=process_block,
        init_val=(initial_O, initial_logsumexp, initial_max),  # , initial_sumV
    )

    # unnecessary if we're using one-hot targets
    # logz_outer = hax.einsum("->...", log_z, sum_v)
    logz_outer = log_z
    o = logz_outer - o

    return o, log_z


def _block_one_hot(LBlock, block_start, labels, dtype):
    end = block_start + LBlock.size
    target_is_in_this_block = hax.logical_and(labels >= block_start, labels < end)
    target_y_block = hax.nn.one_hot(labels - block_start, LBlock, dtype=dtype)
    # 0 out the logits that are not in this block
    target_y_block *= target_is_in_this_block
    return target_y_block


