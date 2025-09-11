import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import haliax as hax
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.service import GenerationService
from levanter.inference.utils import INVALID
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)

@dataclass
class SampleLmConfig:
    """Configuration for simple text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    prompts: list[str] | str | tuple[str, ...] = (
        "Four score and seven years ago, our",
        # "On the first day of Christmas, my true love gave to me",
        "In a hole in the ground there lived a hobbit, not a nasty, dirty, wet hole",
    )
    stop_sequence: str | None = "."
    "Stop sequences. Currently only does whole token sequences."
    max_new_tokens: int = 192
    temperature: float = 0.7
    seed: int = 2

    n_generations: int = 1


def _load_model(config: SampleLmConfig, Vocab: Axis, *, key) -> LmHeadModel:
    """Load a model either from a checkpoint or HF repo."""

    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

    mp = config.trainer.mp

    if config.checkpoint_path is not None:
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = mp.cast_to_compute(model)
        return model
    else:
        assert hasattr(config.model, "hf_checkpoint_converter"), "model config lacks HF loader"
        converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=config.hf_checkpoint,
                                       tokenizer=load_tokenizer(config.tokenizer))
        model = converter.load_pretrained(config.model.model_type, ref=config.hf_checkpoint, dtype=config.trainer.mp.compute_dtype)
        return model


def main(config: SampleLmConfig):
    levanter.initialize(config)
    tok_string: str | None = config.tokenizer
    if config.tokenizer is None:
        if config.hf_checkpoint is not None:
            # If we have an HF checkpoint, we can load the tokenizer from it
            tok_string = config.hf_checkpoint.model_name_or_path

    if tok_string is None:
        raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")

    tokenizer = load_tokenizer(config.tokenizer)

    key = jrandom.PRNGKey(config.seed)

    # NB: we use the compute_axis_mapping b/c we're doing inference
    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        model = _load_model(config, Vocab, key=key)
        assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

        prompts = config.prompts

        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

        # Initialize a reusable generation service with capacity sized to this batch
        max_pages = 64
        max_seqs = len(prompt_ids) * config.n_generations
        page_size = 8
        max_pages_per_seq = 32
        service = GenerationService.from_model(
            model=model,
            tokenizer=tokenizer,
            vocab_axis=Vocab,
            max_pages=max_pages,
            max_seqs=max_seqs,
            page_size=page_size,
            max_pages_per_seq=max_pages_per_seq,
            compute_dtype=config.trainer.mp.compute_dtype,
            max_queued_tokens=32,
            max_seqs_in_prefill=16,
        )

        # -------------------------------- Scheduler-based generation --------------------------------

        stop_sequence = config.stop_sequence
        if stop_sequence is not None:
            stop_ids = tokenizer(stop_sequence, add_special_tokens=False)["input_ids"]
            if len(stop_ids) == 0:
                raise ValueError("Stop sequence must be non-empty")
        else:
            stop_ids = None

        for R in range(10):
            for i, toks in enumerate(prompt_ids):
                print(f"Prompt {i}: {toks}")

            time_in = time.time()
            outputs, total_generated = service.generate(
                prompts=prompt_ids,
                n_generations=config.n_generations,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                seed=config.seed,
                stop_tokens=stop_ids,
                max_tokens_per_round=max_seqs,
                max_rounds=8,
            )
            print(
                f"Round {R} took {time.time() - time_in:.2f} seconds, "
                f"generated {total_generated} tokens in {len(outputs)} sequences."
            )

            # Decode and print outputs
            for seq_id, seq_outputs in enumerate(outputs):
                seq_outputs = [tok for tok in seq_outputs if tok != tokenizer.pad_token_id and tok != INVALID]
                text = tokenizer.decode(seq_outputs, skip_special_tokens=True)
                print(f"Tokens for sequence {seq_id} (len: {len(seq_outputs)}: {seq_outputs}")
                print(f"Generated text for {seq_id}: {text}")
                tokens_with_text = [
                    f"{tok} ({tokenizer.decode([tok], skip_special_tokens=True)})" for tok in seq_outputs
                ]
                print(f"Tokens with text for sequence {seq_id}: {tokens_with_text}")


if __name__ == "__main__":
    levanter.config.main(main)()
