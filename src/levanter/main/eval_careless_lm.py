"""src.levanter.main.eval_careless_lm
===================================
On-the-fly suffix‐likelihood evaluation over a plain-text file.

Compared with ``eval_sliding_lm.py`` this script:
1.  Generates ``LmExample``s directly from a **text cursor** (no cache).
2.  Slides one or more *characters* at a time through the book.
3.  After scoring all suffix windows, produces a *character-level* plot
    of the **maximum** suffix probability covering each character.

It also calls ``levanter.books.util.compute_max_extraction_rates`` to
print the (n, p) discoverability statistics used in memorisation studies.
"""
# NOTE: Do *not* enable postponed evaluation of annotations here because
# levanter.config.main relies on `inspect.getfullargspec(fn).annotations`
# producing real types, not strings.

import itertools
import logging
import math
import pathlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.multihost_utils import process_allgather

from levanter.data import DataLoader, ListAsyncDataset

import haliax as hax
from haliax.nn import log_softmax
from haliax.partitioning import round_axis_for_partitioning

import levanter

# Helpers -----------------------------------------------------------------
from levanter.books.util import compute_max_extraction_rates, create_pz_histogram
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device


class ExampleWithRange(eqx.Module):
    example: LmExample
    char_range: np.ndarray


# -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)


@dataclass
class EvalCarelessLmConfig:
    """YAML/CLI config for the script (mirrors Levanter style)."""

    # Checkpoint options -------------------------------------------------------
    checkpoint_path: Optional[str] = None  # local Levanter checkpoint
    hf_checkpoint: Optional[RepoRef] = None  # e.g. "openai/gpt2"
    initialize_from_hf: Optional[RepoRef] = None
    use_hf_model_config: bool = False

    # Model architecture -------------------------------------------------------
    model: LmConfig = field(default_factory=Gpt2Config)

    # Data ---------------------------------------------------------------------
    txt_path: str | pathlib.Path = "src/levanter/data/books/gatsby.txt"
    chunk_size: int = 100
    slice_length: int = 2000
    prompt_tokens: int = 50
    cursor_inc_chars: int = 10  # stride in characters

    # Tokenizer ----------------------------------------------------------------
    tokenizer_name: Optional[str] = None  # e.g. "meta-llama/Llama-3.1-8B"

    # Runtime / Trainer --------------------------------------------------------
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    max_examples: Optional[int] = None  # cap for quick debug; None → all

    # Output -------------------------------------------------------------------
    plot_path: str = "bar_plot_char_max_pz_70b.png"
    eval_batch_size: int = 32
    histogram_path: str = "pz_distribution_histogram.png"
    pz_threshold: float = 0.0001
    book_title: str = "Book"

    # Performance trade-offs ---------------------------------------------------
    use_dataloader: bool = True
    gather_every_n_batches: int = 1
    """Number of batches to accumulate on device before gathering to host."""


# -----------------------------------------------------------------------------------------
# Main ------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


def main(cfg: EvalCarelessLmConfig):
    levanter.initialize(cfg)

    # Tokenizer & axes ---------------------------------------------------------
    if cfg.tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    else:
        # fall back to model-provided tokenizer (works for GPT-style configs)
        tokenizer = getattr(cfg.model, "the_tokenizer", None)

    if tokenizer is None:
        raise ValueError("Tokenizer not provided: set tokenizer_name in config or ensure model.the_tokenizer exists")

    Pos = cfg.model.Pos
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Build / load model --------------------------------------------------------
    cmapping = cfg.trainer.compute_axis_mapping
    pmapping = cfg.trainer.parameter_axis_mapping

    with cfg.trainer.device_mesh, hax.axis_mapping(pmapping):
        key = jax.random.PRNGKey(0)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), cmapping)
        mp: jmp.Policy = cfg.trainer.mp

        if cfg.checkpoint_path:
            with use_cpu_device():
                model = eqx.filter_eval_shape(cfg.model.build, Vocab, key=key)
                model = load_checkpoint(model, cfg.checkpoint_path, subpath="model")
            model = hax.shard_with_axis_mapping(model, pmapping)
        else:
            hf_ref = cfg.hf_checkpoint or cfg.initialize_from_hf
            if hf_ref is None:
                raise ValueError("Need --checkpoint-path or --hf-checkpoint")
            converter: HFCheckpointConverter = cfg.model.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=hf_ref, tokenizer=tokenizer)
            if cfg.use_hf_model_config:
                cfg.model = converter.config_from_hf_config(converter.default_hf_config)
            model = converter.load_pretrained(cfg.model.model_type, ref=hf_ref, dtype=mp.compute_dtype)

        # Sequence log-prob function -------------------------------------------
        def sequence_log_prob(mod: LmHeadModel, batch: LmExample):
            mod = mp.cast_to_compute(mod)
            with hax.axis_mapping(cmapping):
                logits = mod(batch.tokens, attn_mask=batch.attn_mask)
                lp = log_softmax(logits, axis=mod.Vocab)
                targets = hax.roll(batch.tokens, -1, Pos)
                lp = hax.take(lp, mod.Vocab, targets)
                mask = batch.loss_mask * (targets != pad_id).astype(np.float32)
                return hax.sum(lp * mask, axis=Pos)

        sequence_log_prob = hax.named_jit(sequence_log_prob, out_axis_resources=None)

    # Data stream ---------------------------------------------------------------
    raw_text = pathlib.Path(cfg.txt_path).read_text()

    # Build chunk list once to know total work.
    from levanter.books.util import chunk_text_to_sliding_window_token_chunks

    chunks = chunk_text_to_sliding_window_token_chunks(
        raw_text,
        tokenizer,
        chunk_size=cfg.chunk_size,
        slice_length=cfg.slice_length,
        cursor_inc=cfg.cursor_inc_chars,
    )

    total_chunks = len(chunks)
    print(f"Total sliding windows: {total_chunks}", flush=True)

    def chunk_to_example(chunk):
        ids = chunk["input_ids"]
        if len(ids) < Pos.size:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens_named = hax.named(np.array(ids, dtype=np.int32), Pos)
        ex = LmExample.from_prompt_and_completion(Pos, tokens_named, prompt_length=cfg.prompt_tokens, ignore_id=pad_id)
        rng = np.array([chunk["start_idx"], chunk["end_idx"]], dtype=np.int32)
        return ExampleWithRange(example=ex, char_range=rng)

    batch_size = cfg.eval_batch_size if (cfg.eval_batch_size and cfg.eval_batch_size > 0) else 32

    if cfg.use_dataloader:
        dataset = ListAsyncDataset(chunks, is_complete=True).map(chunk_to_example)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            axis_resources=cfg.trainer.compute_axis_mapping,
            mesh=cfg.trainer.device_mesh,
        )
    else:
        examples_iter = map(chunk_to_example, chunks)

        def loader():
            while True:
                block = list(itertools.islice(examples_iter, batch_size))
                if not block:
                    break
                exs = [e.example for e in block]
                ranges = [e.char_range for e in block]
                B = hax.Axis(cfg.trainer.batch_axis, len(exs))
                tokens_b = hax.stack(B, [e.tokens for e in exs])
                loss_b = hax.stack(B, [e.loss_mask for e in exs])
                batch = LmExample(tokens=tokens_b, loss_mask=loss_b, attn_mask=exs[0].attn_mask)
                ranges = np.stack(ranges, axis=0)
                yield ExampleWithRange(example=batch, char_range=ranges)

    # Evaluation loop -----------------------------------------------------------
    pz_list: List[float] = []
    char_ranges: List[Tuple[int, int]] = []

    total_batches = math.ceil(total_chunks / batch_size)
    buffer: List[jax.Array] = []

    for idx, batch in enumerate(loader):
        if cfg.max_examples and idx * batch_size >= cfg.max_examples:
            break

        batch_ex = batch.example
        ranges = np.array(batch.char_range)

        lp = sequence_log_prob(model, batch_ex).array  # shape (batch,)
        buffer.append(jnp.exp(lp))
        char_ranges.extend([tuple(r) for r in ranges.tolist()])

        if len(buffer) >= cfg.gather_every_n_batches:
            stacked = jnp.concatenate(buffer, axis=0)
            gathered = process_allgather(stacked)
            pz_list.extend(gathered.tolist())
            buffer = []

        done = min((idx + 1) * batch_size, total_chunks)
        pct = 100 * done / total_chunks
        print(f"Batch {idx+1}/{total_batches} – {done}/{total_chunks} windows ({pct:.1f} %)", flush=True)

    if buffer:
        stacked = jnp.concatenate(buffer, axis=0)
        gathered = process_allgather(stacked)
        pz_list.extend(gathered.tolist())

    # Extraction statistics -----------------------------------------------------
    stats = compute_max_extraction_rates(pz_list)
    logger.info("First few (n,p) extraction entries: %s", stats[0][:5])

    # Create and save histogram ------------------------------------------------
    hist_stats = create_pz_histogram(
        pz_list=pz_list, threshold=cfg.pz_threshold, save_path=cfg.histogram_path, book_title=cfg.book_title
    )
    if hist_stats:
        logger.info("P_z histogram statistics: %s", hist_stats)
        # Log the histogram as an artifact
        levanter.tracker.current_tracker().log_artifact(cfg.histogram_path, name=cfg.histogram_path, type="plot")

    # Character-level max-P(z) curve -------------------------------------------
    text_len = len(raw_text)
    char_max = np.zeros(text_len, dtype=np.float32)
    for pz, (c0, c1) in zip(pz_list, char_ranges):
        char_max[c0 : c1 + 1] = np.maximum(char_max[c0 : c1 + 1], pz)

    # ------------------------------------------------------------------
    # Visualization: show a *single-row* heat-map so each character
    # column is a vertical bar whose colour encodes max-P(z).
    # Darker = closer to 1, lighter = closer to 0 (see Blues colormap).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(
        char_max[np.newaxis, :],  # shape (1, text_len)
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    ax.set_title(f"{cfg.book_title}: Maximum per-character probability")
    ax.set_xlabel("Book position (character)")
    ax.set_yticks([])  # Hide y-axis (only one row)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Max. probability")

    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=300)
    levanter.tracker.current_tracker().log_artifact(cfg.plot_path, name=cfg.plot_path, type="plot")

    npy_path = pathlib.Path(cfg.plot_path).with_suffix(".npy")
    np.save(npy_path, char_max)
    levanter.tracker.current_tracker().log_artifact(str(npy_path), name=str(npy_path), type="array")

    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
