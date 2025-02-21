import logging
import os

import jax
import jax.numpy as jnp

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import _get_fs_and_plain_path, load_checkpoint
from levanter.data.text import CausalLmDataset
from levanter.models.lm_model import compute_next_token_loss
from levanter.trainer import Trainer
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


def load_grader_model(
    config,
    grader_model_path: str,
    trainer: Trainer,
):
    logger.info(f"Loading grader model from {grader_model_path}")

    Vocab = round_axis_for_partitioning(Axis("vocab", len(config.data.the_tokenizer)), trainer.parameter_axis_mapping)
    logger.info(f"Created Vocab axis with size {len(config.data.the_tokenizer)}")

    # We'll use the trainer's model config to build the model
    logger.info("Initializing model state...")
    state = trainer.initial_state(
        jax.random.PRNGKey(0), model_init=lambda: config.model.build(Vocab, key=jax.random.PRNGKey(0))
    )

    logger.info("Loading checkpoint...")
    state = load_checkpoint(state, grader_model_path)
    logger.info("Grader model loaded successfully")

    return state.model


def compute_loss_per_sequence(model, example, trainer):
    """Compute per-sequence loss using the same logic as eval.py"""
    model = inference_mode(model, True)
    model = trainer.mp.cast_to_compute(model)

    with hax.axis_mapping(trainer.compute_axis_mapping):
        logger.info("Computing loss per sequence")
        losses = compute_next_token_loss(model, example, reduction=None, reduction_axis=())
        logger.info("Losses computed")
        mask = example.loss_mask  # [Batch, Pos]

        # Sum losses for each sequence in the batch
        per_sequence_loss = hax.sum(losses * mask, axis=model.Pos)  # [Batch]
        per_sequence_tokens = hax.sum(mask, axis=model.Pos)  # [Batch]

        # Normalize by sequence length
        normalized_loss = per_sequence_loss / per_sequence_tokens

    return normalized_loss.array


def grade_datapoints(
    config,
    trainer: Trainer,
    train_dataset: CausalLmDataset,
):
    logger.info("Starting grade_datapoints function")
    grader_model_base_path = (
        "gs://marin-us-central2/checkpoints/suhas/rs-c4-3B-150m-seed{seed}-debug/checkpoints/step-2700"
    )
    grader_model_seeds = [13, 6, 8, 11, 10, 12, 15, 8, 6, 1]
    logger.info(f"Will process {len(grader_model_seeds)} grader models")

    all_grader_losses = []
    for grader_model_seed in grader_model_seeds:
        logger.info(f"\n{'='*50}\nProcessing grader model with seed {grader_model_seed}")
        grader_model_path = grader_model_base_path.format(seed=grader_model_seed)

        grader_model = load_grader_model(config, grader_model_path, trainer)
        grader_model = inference_mode(grader_model, True)
        logger.info("Grader model loaded and set to inference mode")

        train_loader = trainer.data_loader(train_dataset)
        logger.info("Data loader created successfully")

        iter_data = iter(train_loader)

        step = 0
        grader_losses = []
        total_sequences = 0

        logger.info(f"Beginning evaluation loop (will run for {trainer.num_train_steps} steps)")
        while step < trainer.num_train_steps:
            if step % 500 == 0:
                logger.info(f"Processing step {step}/{trainer.num_train_steps}")

            example = next(iter_data)
            batch_losses = compute_loss_per_sequence(grader_model, example, trainer)

            # Track statistics
            grader_losses.append(batch_losses)
            total_sequences += batch_losses.shape[0]

            levanter.tracker.log({"grader_eval/num_sequences": total_sequences}, step=step)

            step += 1

        # Compute final statistics
        grader_losses_array = jnp.concatenate(grader_losses)
        final_mean = float(grader_losses_array.mean())
        final_std = float(grader_losses_array.std())

        logger.info(f"Grader model {grader_model_seed} finished grading all datapoints")
        logger.info("Final statistics:")
        logger.info(f"  Total steps processed: {step}")
        logger.info(f"  Final average loss: {final_mean:.4f}")
        logger.info(f"  Loss standard deviation: {final_std:.4f}")
        logger.info(f"  Total sequences evaluated: {total_sequences}")

        del train_loader
        del grader_model
        logger.info("Resources cleaned up")
        all_grader_losses.append(grader_losses_array)

    all_grader_losses = jnp.array(all_grader_losses)  # shape [num_seeds, num_sequences]
    logger.info(f"All grader losses shape: {all_grader_losses.shape}")  # type: ignore

    # Compute variance along the seeds axis (axis=0)
    loss_variances = jnp.var(all_grader_losses, axis=0)

    # Print the array of variances
    logger.info("Variance of losses across seeds for each sequence:")
    logger.info(loss_variances)

    # Compute the mean of the variances
    mean_variance = jnp.mean(loss_variances)
    logger.info(f"Mean variance of losses across seeds for each sequence: {mean_variance:.4f}")

    # Compute the standard deviation of the variances
    std_variance = jnp.std(loss_variances)
    logger.info(f"Standard deviation of losses across seeds for each sequence: {std_variance:.4f}")

    # Saving loss variances to a file
    fs, plain_path = _get_fs_and_plain_path("gs://marin-us-central2/scratch/suhas/loss_arrays/")
    fs.makedirs(plain_path, exist_ok=True)

    loss_variances_path = os.path.join(plain_path, "loss_variances.npy")
    all_grader_losses_path = os.path.join(plain_path, "all_grader_losses.npy")

    # Only save on process 0 to avoid conflicts
    if jax.process_index() == 0:
        with fs.open(loss_variances_path, "wb") as f:
            jnp.save(f, loss_variances)
        logger.info(f"Loss variances saved to {loss_variances_path}")

        with fs.open(all_grader_losses_path, "wb") as f:
            jnp.save(f, all_grader_losses)
        logger.info(f"All grader losses saved to {all_grader_losses_path}")

    logger.info("Grade_datapoints function completed successfully")
