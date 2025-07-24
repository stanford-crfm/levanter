import dataclasses

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
from levanter.inference.jit_scheduler import JitScheduler, PackedSequence
from levanter.layers.attention import KvPageCache
from jaxtyping import PRNGKeyArray

logger = logging.getLogger(__name__)


class GenState(eqx.Module):
    """
    Plain Old Data type for generation state.
    Contains all the components needed for language model generation.
    """
    sched: JitScheduler
    cache: KvPageCache
    page_table: PageTable
    prng_key: PRNGKeyArray


@dataclass
class SampleLmConfig:
    """Configuration for simple text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    prompts: list[str] | str | tuple[str, ...] = ("Four score and seven years ago, our", "Now is the time for")
    max_new_tokens: int = 32
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


@haliax.named_jit(donate_args=(True, False, False))
def run_generation_loop(
    gen_state: GenState,
    model,
    sampler,
    temps,
    max_tokens_per_round: int,
    max_rounds: int,
) -> tuple[GenState, PackedSequence]:
    """Generate tokens using ``JitScheduler`` until either ``max_new_tokens`` have been
    produced *per sequence* or all sequences report finished."""

    def cond(state: tuple[GenState, jax.Array]):
        _gen_state, step = state
        return (step < max_rounds) & (_gen_state.sched.num_queued_tokens > 0) & (_gen_state.sched.empty_generated_space > 0)

    def body(state):
        gen_state: GenState
        gen_state, step = state

        # Pack the next chunk from the queue
        sched, packed_seq = gen_state.sched.pack_next_sequence(max_tokens_per_round)

        page_table, binfo = gen_state.page_table.allocate_for_seq(token_seq_ids=packed_seq.seq_ids)

        # Decode logits and sample new tokens
        logits, cache = model.decode(packed_seq.tokens, gen_state.cache, binfo, binfo.pos_ids)
        sample_key, key = jrandom.split(gen_state.prng_key)
        boundaries = packed_seq.boundary_indices(page_table.max_seqs)
        # jax.debug.print("Boundaries: {boundaries} {ids} {toks} {bound2}", boundaries=boundaries, ids=packed_seq.seq_ids, toks=packed_seq.tokens, bound2=packed_seq.is_boundary)
        logits = logits["position", boundaries]
        new_tokens, _ = sampler(logits, temps, key=sample_key)

        num_new_tokens = hax.sum(boundaries != -1).scalar().astype(jnp.int32)

        # Update scheduler with the freshly sampled tokens
        sched = sched.update_after_sampling(
            new_tokens=new_tokens,
            new_token_seq_ids=packed_seq.seq_ids["position", boundaries],
            num_new_tokens=num_new_tokens,
        )

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(
            gen_state,
            sched=sched,
            page_table=page_table,
            cache=cache,
            prng_key=key
        )

        return new_gen_state, step + 1

    init_state = (gen_state, jnp.array(0, dtype=jnp.int32))
    final_gen_state, _ = jax.lax.while_loop(cond, body, init_state)

    # Extract the packed sequence from the scheduler
    sched, out_seq = final_gen_state.sched.extract_all_generated_tokens()

    final_gen_state = dataclasses.replace(final_gen_state, sched=sched)

    return final_gen_state, out_seq


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

        prompts = config.prompts

        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

        table = PageTable.init(64, len(prompt_ids), 16, 4)
        kv_cache = model.initial_cache(table, dtype=config.trainer.mp.compute_dtype)

        # Initialize GenState with all components
        gen_state = GenState(
            sched=JitScheduler.init(64, 32),
            cache=kv_cache,
            page_table=table,
            prng_key=jrandom.PRNGKey(0)
        )

        temps = hax.full((), config.temperature, dtype=jnp.float32)

        total_prompt_tokens = sum(len(tokens) for tokens in prompt_ids)

        MAX_NEW_TOKENS = 32

        # -------------------------------- Scheduler-based generation --------------------------------
        for R in range(10):
            if R == 10:
                start_profiler("/tmp/gen-profile")
            elif R == 20:
                stop_profiler_and_maybe_wait(create_perfetto_link=False)
                levanter.current_tracker().log_artifact("/tmp/gen-profile", type="jax_profile")
            time_in = time.time()

            # Reset gen_state for this round
            import dataclasses
            gen_state = dataclasses.replace(gen_state, prng_key=jrandom.PRNGKey(0))
            gen_state = dataclasses.replace(gen_state, page_table=gen_state.page_table.free_pages(0))

            finished = [False] * len(prompt_ids)

            outputs = [list(t) for t in prompt_ids]  # start with the prompts

            # do one prefill at a time, but we do continuous batching, so decode is happening all the while
            for tokens in prompt_ids:
                # enqueue the entire prompt into the scheduler
                page_table, seq_id = gen_state.page_table.assign_seq_id_to_seq()
                gen_state = dataclasses.replace(gen_state, page_table=page_table)

                prompts_tokens_to_enqueue = np.asarray(tokens, dtype=jnp.int32)
                if len(tokens) > gen_state.sched.max_queued_tokens:
                    raise ValueError(
                        f"Prompt is too long ({len(tokens)} tokens), "
                        f"max allowed is {MAX_NEW_TOKENS} tokens."
                    )

                target_len = len(prompts_tokens_to_enqueue)
                if target_len > gen_state.sched.max_queued_tokens:
                    print(f"Queue is full ({gen_state.sched.num_queued_tokens} tokens), running generation loop to free up space.")
                while target_len > gen_state.sched.empty_queue_space:
                    # if the queue is too full, we run generation loop to free up space
                    # TODO: would be better if we do partial/chunked prefill here, but hopefully this is rare
                    gen_state, packed_out = run_generation_loop(
                        gen_state,
                        model,
                        sampler,
                        temps,
                        # TODO: tune/configure
                        64,
                        32,
                    )

                    extract_outputs(packed_out, outputs, finished, total_prompt_tokens, config.max_new_tokens)

                this_tokens = hax.named(prompts_tokens_to_enqueue, axis="position")
                seq_ids = hax.full_like(this_tokens, seq_id, dtype=jnp.int32)
                new_sched = gen_state.sched.enqueue_tokens(this_tokens, seq_ids, prompts_tokens_to_enqueue.size)
                gen_state = dataclasses.replace(gen_state, sched=new_sched)
                del new_sched

                # do one macro-prefill round
                gen_state, packed_out = run_generation_loop(
                    gen_state,
                    model,
                    sampler,
                    temps,
                    16,
                    1,
                )

                extract_outputs(packed_out, outputs, finished, total_prompt_tokens, config.max_new_tokens)

            gen_state = jax.block_until_ready(gen_state)
            time_mid = time.time()
            print(f"Prefill took {time_mid - time_in:.2f} seconds, ")

            # run the fully JIT-compiled generation loop
            while not all(finished):
                # if the queue is too full, we run generation loop to free up space
                # TODO: would be better if we do partial/chunked prefill here, but hopefully this is rare
                gen_state, this_outputs = run_generation_loop(
                    gen_state,
                    model,
                    sampler,
                    temps,
                    # TODO: tune/configure
                    len(prompt_ids),
                    32,
                )

                extract_outputs(this_outputs, outputs, finished, total_prompt_tokens, config.max_new_tokens)

            gen_state = jax.block_until_ready(gen_state)
            time_out = time.time()
            print(f"Gen loop took {time_out - time_mid:.2f} seconds, ")


            # Flatten, drop padding, and decode
            total_generated = sum(len(seq_outputs) for seq_outputs in outputs)
            total_generated -= sum(len(p) for p in prompt_ids)  # remove prompt tokens
            for seq_id, seq_outputs in enumerate(outputs):
                if finished[seq_id]:
                    # remove padding tokens
                    seq_outputs = [tok for tok in seq_outputs if tok != tokenizer.pad_token_id and tok >= 0]
                    outputs[seq_id] = seq_outputs
                else:
                    print(f"Sequence {seq_id} did not finish, skipping decoding.")

                text = tokenizer.decode(seq_outputs, skip_special_tokens=True)
                print(f"Generated text for {seq_id}: {text}")
            print(f"Round {R} took {time.time() - time_in:.2f} seconds, "
                  f"generated {total_generated} tokens in {len(outputs)} sequences.")
            # free everything
            page_table = PageTable.init(64, len(prompt_ids), 16, 4)
            kv_cache = model.initial_cache(page_table, dtype=config.trainer.mp.compute_dtype)
            gen_state = dataclasses.replace(gen_state, page_table=page_table, cache=kv_cache)


# @haliax.named_jit(donate_args=(True,))
def extract_outputs(out_seq: PackedSequence, outputs, finished, total_prompt_tokens, max_new_tokens):
    """
    drain generated tokens and append them to the outputs.

    MUTATES outputs and finished lists.
    """
    # TODO: offload this and detokenization to a separate thread
    out_seq = jax.device_get(out_seq)
    for i in range(out_seq.num_tokens):
        seq_id = int(out_seq.seq_ids.array[i])
        tok_id = int(out_seq.tokens.array[i])

        if seq_id >= len(outputs) or seq_id < 0:
            continue
        if finished[seq_id]:
            continue
        outputs[seq_id].append(tok_id)

        if len(outputs[seq_id]) >= max_new_tokens + total_prompt_tokens:
            finished[seq_id] = True


if __name__ == "__main__":
    levanter.config.main(main)()
