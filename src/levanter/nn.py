from functools import partial
from typing import Callable, Dict, Tuple

import haliax as hax
from haliax import Axis, NamedArray
from haliax import nn as hnn


def cross_entropy_loss(
    pred_y: NamedArray,
    Label: Axis,
    target_y: NamedArray,
) -> NamedArray:
    loss, _ = cross_entropy_loss_and_log_normalizers(pred_y, Label, target_y)
    return loss


def cross_entropy_loss_and_log_normalizers(
    pred_y: NamedArray,
    Label: Axis,
    target_y: NamedArray,
) -> Tuple[NamedArray, NamedArray]:
    """
    Compute the cross entropy loss and log normalizers for a batch of predictions and targets.

    :param pred_y: a NamedArray with the Label axis (and possibly others for e.g. batch and seq) containing the logits
    :param Label: the Label axis
    :param target_y: a NamedArray with the Label axis (and possibly others) containing the targets

    :return: tuple of two named arrays, with "per position" losses and log normalizers
    """
    log_normalizers = hax.nn.logsumexp(pred_y, Label)
    neg_log_normalized = log_normalizers - pred_y

    loss = hax.dot(Label, target_y, neg_log_normalized)

    return loss, log_normalizers


def quick_gelu(x):
    return x * hnn.sigmoid(1.702 * x)


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": quick_gelu,
}
