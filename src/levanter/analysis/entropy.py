"""Functions for computing and visualizing token-level entropy."""
import logging
from typing import Callable, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PyTree

import haliax as hax
import haliax.nn as hnn

import levanter.tracker

from ..callbacks import StepInfo
from ..data import DataLoader
from ..tracker.histogram import Histogram


B = TypeVar("B")

logger = logging.getLogger(__name__)


def entropy_from_logits(logits: hax.NamedArray, axis: hax.AxisSelector) -> hax.NamedArray:
    """
    Computes entropy over the given axis in a numerically stable way using raw logits.

    Entropy = logsumexp(logits) - sum(softmax(logits) * logits)

    Args:
        logits: A NamedArray of logits.
        axis: The axis over which to compute the entropy (e.g. Vocabulary axis).

    Returns:
        A NamedArray of entropies, with `axis` removed.
    """
    log_z = hnn.logsumexp(logits, axis=axis)
    probs = hax.exp(logits - log_z)
    entropy = log_z - hax.sum(probs * logits, axis=axis)
    return entropy


def compute_entropy_histogram(
    model,
    Vocab: hax.AxisSelector,
    logit_fn: Callable[[PyTree, B], hax.NamedArray],
    test_data,
    max_tokens: int = 1024 * 1024,
    num_bins: int = 64,
) -> Histogram:
    """
    Compute entropy distribution for a given model and dataset.

    Args:
        model: Model to use for computing logits
        logit_fn: Function that takes (model, batch) and returns logits
        test_data: Test data to use for computing entropy
        max_tokens: Maximum number of tokens to process
        num_bins: Number of bins for the histogram

    Returns:
        Histogram object containing the entropy distribution
    """

    entropies_list: list[jnp.ndarray] = []
    total_tokens = 0

    for batch in test_data:
        batch_entropies = _compute_entropy_on_device(logit_fn, model, batch, Vocab)
        entropies_list.append(batch_entropies)
        total_tokens += batch_entropies.size

        if total_tokens >= max_tokens:
            break

    entropies = jnp.concatenate(entropies_list)

    if not entropies.size:
        raise ValueError("No tokens processed")

    # Create and return histogram
    return Histogram.from_array(entropies, num_bins=num_bins)


# Top level to avoid recompilation
@eqx.filter_jit
def _compute_entropy_on_device(logit_fn, model, batch: B, Vocab) -> jnp.ndarray:
    logits = logit_fn(model, batch)
    entropies = entropy_from_logits(logits, axis=Vocab)
    # flatten out batch etc so it's fully replicated
    return entropies.flatten("token").array


def cb_compute_entropies(
    logit_fn,
    Vocab: hax.AxisSelector,
    test_data,
    key: str | None,
    batch_size: int,
    mapping: hax.partitioning.ResourceMapping,
    num_tokens: int = 1024 * 1024,
):
    """
    Callback to compute entropy distribution and log it to the tracker.

    Args:
        logit_fn: Function that takes (model, batch) and returns logits
        Vocab (hax.AxisSelector): The vocabulary to use.
        test_data: The test data to use.
        key (str | None): The key to log to the tracker. If None, "entropy" is used.
        num_tokens: The number of tokens to use.
        batch_size: The batch size to use.
        mapping: The resource mapping

    Returns:
        function: A function that takes a step info and computes and visualizes the log probabilities.
    """
    if key is None:
        key = "entropy"

    def compute_entropy(step: StepInfo):
        data_loader = DataLoader(test_data, batch_size=batch_size, pad_final_batch=False, axis_resources=mapping)
        model = step.eval_model

        try:
            hist = compute_entropy_histogram(
                model=model,
                Vocab=Vocab,
                logit_fn=logit_fn,
                test_data=data_loader,
                max_tokens=num_tokens,
            )
        except ValueError as e:
            if "No tokens processed" in str(e):
                logger.warning(f"{key} is too small to compute entropy with batch size {batch_size}")
                return
            logger.exception(f"Error computing entropy for {key}")
            raise

        levanter.tracker.log({key: hist}, step=step.step)

    return compute_entropy
