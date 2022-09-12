import logging
from typing import Callable, Iterator, TypeVar

import jax.numpy as jnp
from tqdm import tqdm

import wandb
from levanter.checkpoint import save_checkpoint
from levanter.modeling_utils import RunningMean
from levanter.trainer_hooks import StepInfo


logger = logging.getLogger(__name__)


def save_model(run_dir, prepare_fn=None):
    if not prepare_fn:
        prepare_fn = lambda x: x  # noqa F731

    def save(info: StepInfo):
        # TODO: when we do multi-machine model sharding we should do something cleverer.
        # it's actually pretty easy to save the model and the optimizer state
        # and enable resuming
        if info.step != 0:
            save_checkpoint(
                model=prepare_fn(info.model),
                training_state=((prepare_fn(info.opt_state)), info.next_key),
                step=info.step,
                checkpoint_path=f"{run_dir}/step-{info.step}",
            )

    return save


M = TypeVar("M")
X = TypeVar("X")
Y = TypeVar("Y")


def compute_validation_loss(
    loss_fn: Callable,  # [[M, ...], jax.numpy.ndarray],
    dataloader: Callable[[], Iterator[tuple]],
):
    def compute_loss(info: StepInfo):
        total_loss = RunningMean(shape=1)
        test_loader = dataloader()

        pbar = tqdm(test_loader, desc="eval", position=1, leave=False)
        for batch in pbar:
            loss = loss_fn(info.model, *batch)
            # this mean is over the devices, somewhat confusingly
            loss = jnp.mean(loss)
            total_loss.update(loss)
            pbar.set_postfix(loss=total_loss.mean.item())

        mean_loss = total_loss.mean.item()
        if wandb.run is not None:
            wandb.log({"eval/loss": mean_loss}, step=info.step)

        logger.info(f"validation loss: {mean_loss:.3f}")

        return total_loss

    return compute_loss
