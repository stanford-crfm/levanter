from typing import Callable, TypeVar, Tuple, Iterator

import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
from tqdm import tqdm

from psithuros.modeling_utils import RunningMean
from psithuros.trainer_hooks import StepInfo
from psithuros.checkpoint import save_checkpoint
from psithuros import jax_utils


def get_nth_rank(pytree, rank=0, leaf_filter=eqx.is_inexact_array):
    return jax.tree_map(lambda leaf: leaf[rank] if leaf_filter(leaf) else leaf, pytree)


def save_model(run_dir, prepare_fn=None):

    if not prepare_fn:
        prepare_fn = lambda x: x

    def save(info: StepInfo):
        # TODO: when we do model sharding we have to do something cleverer
        # it's actually pretty easy to save the model and the optimizer state
        # and enable resuming
        save_checkpoint(model=prepare_fn(info.model),
                        training_state=((prepare_fn(info.opt_state)), info.next_key),
                        step=info.step,
                        checkpoint_path=f"{run_dir}/step-{info.step}")

    return save

M = TypeVar("M")
X = TypeVar("X")
Y = TypeVar("Y")

def compute_validation_loss(loss_fn: Callable[[M, X, Y], jax.numpy.ndarray],
                            dataloader: Callable[[], Iterator[Tuple[X, Y]]]):
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

        total_loss = total_loss.mean.item()
        wandb.log({"eval/loss": total_loss}, step=info.step)


    return compute_loss
