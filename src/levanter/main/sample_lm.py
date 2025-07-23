import numpy as np
import time

import jax

import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from haliax.jax_utils import is_jax_array_like

from levanter.callbacks import start_profiler, stop_profiler_and_maybe_wait
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.layers.page_table import PageTable
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.inference.jit_scheduler import JitScheduler

logger = logging.getLogger(__name__)


@dataclass
class SampleLmConfig:
    """Configuration for simple text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    prompt: str = "Four score and seven years ago, our"
    max_new_tokens: int = 100
    temperature: float = 1e-4


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


def tree_byte_size(tree):
    """Calculate the total byte size of a JAX tree."""

    # TODO: take into account sharding
    def _leaf_size(x):
        if is_jax_array_like(x):
            return x.nbytes
        return 0

    return sum(_leaf_size(x) for x in jax.tree.leaves(tree))


@haliax.named_jit(donate_args=(True, True, True))
def run_generation_loop(
    sched: JitScheduler,
    page_table: PageTable,
    cache,
    model,
    sampler,
    temps,
    key,
    max_tokens_per_round: int,
    max_new_tokens: int,
):
    """Generate tokens using ``JitScheduler`` until either ``max_new_tokens`` have been
    produced *per sequence* or all sequences report finished."""

    def cond(state):
        _sched: JitScheduler
        _sched, *_ , step = state
        return (step < max_new_tokens) & (_sched.num_queued_tokens > 0)

    def body(state):
        sched: JitScheduler
        sched, page_table, cache, key, step = state

        # Pack the next chunk from the queue
        sched, packed_seq = sched.pack_next_sequence(max_tokens_per_round)

        page_table, binfo = page_table.allocate_for_seq(token_seq_ids=packed_seq.seq_ids)

        # Decode logits and sample new tokens
        logits, cache = model.decode(packed_seq.tokens, cache, binfo, binfo.pos_ids)
        sample_key, key = jrandom.split(key)
        boundaries = packed_seq.boundary_indices(page_table.max_seqs)
        # jax.debug.print("Boundaries: {boundaries} {ids} {toks} {bound2}", boundaries=boundaries, ids=packed_seq.seq_ids, toks=packed_seq.tokens, bound2=packed_seq.is_boundary)
        logits = logits["position", boundaries]
        new_tokens, _ = sampler(logits, temps, key=sample_key)

        num_new_tokens = hax.sum(boundaries != -1).scalar()

        # Update scheduler with the freshly sampled tokens
        sched = sched.update_after_sampling(
            new_tokens=new_tokens,
            new_token_seq_ids=packed_seq.seq_ids["position", boundaries],
            num_new_tokens=num_new_tokens,
        )
        return sched, page_table, cache, key, step + 1

    init_state = (sched, page_table, cache, key, jnp.array(0, dtype=jnp.int32))
    sched, page_table, cache, key, _ = jax.lax.while_loop(cond, body, init_state)
    return sched, cache, page_table, key


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

    key = jrandom.PRNGKey(0)

    # NB: we use the compute_axis_mapping b/c we're doing inference
    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        model = _load_model(config, Vocab, key=key)
        assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

        sampler = Sampler(Vocab)

        prompt_ids = tokenizer.encode(config.prompt, add_special_tokens=False)
        # print(f"Prompt: {config.prompt}")
        # print(f"Tokens: {prompt_ids}")

        page_table = PageTable.init(
            max_pages=4,
            max_seqs=1,
            page_size=16,
            max_pages_per_seq=4,
        )
        cache = eqx.filter_jit(model.initial_cache)(page_table, dtype=config.trainer.mp.compute_dtype)
        cache = hax.auto_sharded(cache)

        temps = hax.full((), config.temperature, dtype=jnp.float32)

        MAX_TOKENS = 32    # per‐round chunk size
        MAX_SEQS = 1      # hot‐set size, for this we're only decoding 1 at a time so

        # -------------------------------- Scheduler-based generation --------------------------------
        for R in range(10):
            if R == 10:
                start_profiler("/tmp/gen-profile")
            elif R == 20:
                stop_profiler_and_maybe_wait(create_perfetto_link=False)
                levanter.current_tracker().log_artifact("/tmp/gen-profile", type="jax_profile")
            time_in = time.time()
            prng_key = jrandom.PRNGKey(0)
            page_table = page_table.free_pages(0)
            page_table, seq_id = page_table.assign_seq_id_to_seq()

            sched = JitScheduler.init(max_tokens=MAX_TOKENS, max_seqs=MAX_SEQS, key=key)

            # enqueue the entire prompt into the scheduler
            prompts_tokens_to_enqueue = np.asarray(prompt_ids, dtype=jnp.int32)
            while len(prompts_tokens_to_enqueue):
                next_queue = prompts_tokens_to_enqueue[:MAX_TOKENS]
                prompts_tokens_to_enqueue = prompts_tokens_to_enqueue[MAX_TOKENS:]
                this_tokens = hax.named(next_queue, axis="position")
                seq_ids = hax.full_like(this_tokens, seq_id, dtype=jnp.int32)
                sched = sched.enqueue_tokens(this_tokens, seq_ids, next_queue.size)

                # do one macro-prefill round
                sched, cache, page_table, prng_key = run_generation_loop(
                    sched,
                    page_table,
                    cache,
                    model,
                    sampler,
                    temps,
                    prng_key,
                    next_queue.size,
                    1,
                )

            sched = jax.block_until_ready(sched)
            time_mid = time.time()
            print(f"Prefill took {time_mid - time_in:.2f} seconds, ")

            # run the fully JIT-compiled generation loop
            sched, cache, page_table, prng_key = run_generation_loop(
                sched,
                page_table,
                cache,
                model,
                sampler,
                temps,
                prng_key,
                1,
                min(config.max_new_tokens, MAX_TOKENS),
            )
            sched = jax.block_until_ready(sched)
            time_out = time.time()
            print(f"Gen loop took {time_out - time_mid:.2f} seconds, ")

            # extract up to `max_new_tokens` tokens for this sequence
            out_ids = hax.named(jnp.array([seq_id], dtype=jnp.int32), axis="seq")
            sched, output_matrix = sched.extract_generated_tokens(out_ids, max_tokens=config.max_new_tokens)

            # Flatten, drop padding, and decode
            generated_token_ids = [int(t) for t in prompt_ids]
            generated_token_ids.extend([int(tok) for tok in output_matrix.array[0] if tok != -1])
            text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            print(f"Generated text: {text}")
            print(f"Round {R} took {time.time() - time_in:.2f} seconds, "
                  f"generated {len(generated_token_ids) - len(prompt_ids)} tokens, ")
            page_table = page_table.free_pages(0)

        return text



if __name__ == "__main__":
    levanter.config.main(main)()
