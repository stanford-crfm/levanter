import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jmp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.multihost_utils import process_allgather

import haliax as hax
from haliax import Axis
from haliax.nn import log_softmax
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfig, UrlSingleDatasetLMConfig
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device


# Visualization tweak flags (set to True/False as desired)
WIDE_LINES: bool = False  # If True, draw thicker vertical bars in the plot
RESCALE_COLORMAP: bool = True  # If True, rescale color map to the data's min/max probabilities

# Dataset verification flag – set to True if you want to sanity-check that each example
# really has a 50-token prompt followed by a 50-token response (assuming Pos.size >= 100)
VERIFY_PROMPT_RESP_LEN: bool = False
EXPECTED_PROMPT_TOKENS = 50
EXPECTED_RESPONSE_TOKENS = 50

logger = logging.getLogger(__name__)


@dataclass
class EvalSlidingLmConfig:
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    initialize_from_hf: Optional[RepoRef] = None
    """HF checkpoint to load for evaluation."""
    use_hf_model_config: bool = False
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: SingleDatasetLMConfig | LMMixtureDatasetConfig = field(default_factory=UrlSingleDatasetLMConfig)
    model: LmConfig = field(default_factory=Gpt2Config)

    split: str = "validation"
    max_batches: Optional[int] = None


def main(config: EvalSlidingLmConfig):
    levanter.initialize(config)
    tokenizer = config.data.the_tokenizer

    Pos = config.model.Pos

    cache = config.data.build_or_load_cache(config.split)
    if cache is None:
        raise ValueError(f"No dataset found for split {config.split}")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def _to_example(row):
        ids = row["input_ids"].tolist()
        src_len = int(row["sources_len"])
        if len(ids) > Pos.size:
            ids = ids[: Pos.size]
        else:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens = hax.named(np.array(ids, dtype=np.int32), Pos)
        return LmExample.from_prompt_and_completion(Pos, tokens, prompt_length=src_len)

    dataset = cache.map(_to_example)

    loader = DataLoader(
        dataset,
        batch_size=config.trainer.eval_batch_size,
        axis_resources=config.trainer.compute_axis_mapping,
        mesh=config.trainer.device_mesh,
    )

    if config.max_batches is not None:
        loader = itertools.islice(loader, config.max_batches)

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    hf_ref = config.hf_checkpoint or config.initialize_from_hf

    if config.checkpoint_path is None and hf_ref is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if config.checkpoint_path is not None and hf_ref is not None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint, not both")

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        mp: jmp.Policy = config.trainer.mp

        def compute_sequence_log_prob(model: LmHeadModel, batch: LmExample):
            """
            Computes the log probability of the suffix of each example in the batch.
            """
            model = mp.cast_to_compute(model)
            with hax.axis_mapping(compute_axis_mapping):
                logits = model(batch.tokens, attn_mask=batch.attn_mask)
                lp = log_softmax(logits, axis=model.Vocab)
                targets = hax.roll(batch.tokens, -1, Pos)
                lp = hax.take(lp, model.Vocab, targets)

                # 1. The loss_mask, which masks out the prefix (prompt)
                # 2. A mask for the last token, which we can't predict
                # 3. A mask for padding tokens

                # The loss_mask from the batch is for training, which is slightly off for our purposes.
                # We want to score the logprob of each token in the suffix, including the first and last.
                # The prediction for the first token of the suffix occurs at the last token of the prompt.
                prompt_lengths = hax.sum(1.0 - batch.loss_mask, axis=Pos)
                positions = hax.arange(Pos)

                # True for positions where we are predicting a suffix token.
                # We have to explicitly broadcast prompt_lengths to have the Pos axis for the comparison.
                suffix_mask = positions >= (prompt_lengths - 1).broadcast_axis(Pos)

                padding_mask = (targets != pad_id).astype(np.float32)
                mask = suffix_mask * padding_mask

                # apply the mask and sum over the sequence positions to get total logprob per example
                masked_lp = lp * mask
                total_log_prob = hax.sum(masked_lp, axis=Pos)

                return total_log_prob

        compute_sequence_log_prob = hax.named_jit(compute_sequence_log_prob, out_axis_resources=None)

        if config.checkpoint_path is not None:
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)
        elif hf_ref is not None:
            model_config = config.model
            if not hasattr(model_config, "hf_checkpoint_converter"):
                raise ValueError("Model config does not have an HF checkpoint converter.")
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=hf_ref, tokenizer=tokenizer)
            if config.use_hf_model_config:
                config.model = converter.config_from_hf_config(converter.default_hf_config)
                model_config = config.model
            model = converter.load_pretrained(model_config.model_type, ref=hf_ref, dtype=mp.compute_dtype)
        else:
            raise ValueError("Must specify checkpoint_path or hf_checkpoint")

        all_probs = []
        # helper stats if verification is enabled
        if VERIFY_PROMPT_RESP_LEN:
            checked_batches = 0  # we only want to dump a couple of batches, not the whole run

        for batch in loader:
            if VERIFY_PROMPT_RESP_LEN and checked_batches < 1:  # print the first few batches only
                # prompt length ≈ (# tokens where loss_mask == 0) + 1 (because first suffix token is predicted at last prompt token)
                # Use haliax operations first, then convert to numpy
                prompt_lens_hax = hax.sum(1.0 - batch.loss_mask, axis=Pos) + 1

                # For suffix length, exclude padding tokens
                non_padding_mask = (batch.tokens != pad_id).astype(np.float32)
                suffix_lens_hax = hax.sum(batch.loss_mask * non_padding_mask, axis=Pos)

                # Convert to numpy arrays using process_allgather to handle distributed arrays
                prompt_lens = process_allgather(prompt_lens_hax.array)
                suffix_lens = process_allgather(suffix_lens_hax.array)

                print("==== Dataset length check ====", flush=True)
                print("Prompt lengths:", prompt_lens.astype(int).tolist(), flush=True)
                print("Suffix lengths:", suffix_lens.astype(int).tolist(), flush=True)

                if np.all(prompt_lens == EXPECTED_PROMPT_TOKENS):
                    print("All prompts have expected length", flush=True)
                else:
                    print("⚠️  Mismatch in prompt lengths!", flush=True)

                if np.all(suffix_lens == EXPECTED_RESPONSE_TOKENS):
                    print("All responses have expected length", flush=True)
                else:
                    print("⚠️  Mismatch in response lengths!", flush=True)

                checked_batches += 1

            log_probs = compute_sequence_log_prob(model, batch).array
            log_probs = process_allgather(log_probs)

            probs = np.exp(log_probs)
            all_probs.append(probs)

        if not all_probs:
            raise ValueError("No data processed")

        prob_dist = np.concatenate(all_probs, axis=0)

        # Print the largest probability observed across all examples (helpful for quick sanity-checks)
        max_prob = float(np.max(prob_dist))
        mean_prob = float(np.mean(prob_dist))
        median_prob = float(np.median(prob_dist))
        print(f"Max suffix probability: {max_prob:.6f}", flush=True)
        print(f"Mean suffix probability: {mean_prob:.6f}", flush=True)
        print(f"Median suffix probability: {median_prob:.6f}", flush=True)

        fig, ax = plt.subplots(figsize=(10, 4))  # Adjusted for a barcode-like plot
        example_indices = np.arange(len(prob_dist))

        # Create a colormap where high probability is dark (black) and low is light (white)
        if RESCALE_COLORMAP:
            norm = mcolors.Normalize(vmin=float(prob_dist.min()), vmax=float(prob_dist.max()))
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1.0)
        cmap = plt.get_cmap("Greys")

        # Choose line width based on flag
        line_width = 1.5 if WIDE_LINES else 0.5

        # Plot the vertical lines, all with the same height (spanning 0 to 1)
        # The color of each line is determined by its probability
        ax.vlines(example_indices, 0, 1, colors=cmap(norm(prob_dist)), alpha=0.75, linewidth=line_width)

        # Add a colorbar to serve as a legend for the probabilities
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable, ax=ax)
        cbar.set_label("Probability of Suffix")

        ax.set_xlabel("Example Index")
        ax.set_yticks([])  # Remove y-axis ticks as the height is constant
        ax.set_ylabel("")  # Remove y-axis label
        ax.set_title("Likelihood of Suffix per Example")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(prob_dist))
        plt.tight_layout()
        path = "suffix_likelihood_barcode-v3.png"
        fig.savefig(path)
        levanter.tracker.current_tracker().log_artifact(path, name=path, type="plot")

    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
