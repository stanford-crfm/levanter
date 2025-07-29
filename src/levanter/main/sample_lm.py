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
import haliax.haxtyping as ht
from haliax import Axis, NamedArray
from haliax.partitioning import round_axis_for_partitioning

import levanter
from haliax.jax_utils import is_jax_array_like

from levanter.callbacks import start_profiler, stop_profiler_and_maybe_wait
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.page_table import PageTable
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.inference.jit_scheduler import JitScheduler, DecodeState, SeqDecodingParams
from levanter.inference.utils import INVALID, is_invalid
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
    decode_state: DecodeState
    prng_key: PRNGKeyArray


@dataclass
class SampleLmConfig:
    """Configuration for simple text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    prompts: list[str] | str | tuple[str, ...] = (
        # "Four score and seven years ago, our",
        # "On the first day of Christmas, my true love gave to me",
        "In a hole in the ground there lived a hobbit, not a nasty, dirty, wet hole",
    )
    max_new_tokens: int = 192
    temperature: float = 0.7


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
) -> tuple[GenState, ht.i32[NamedArray, "seq position"], ht.i32[NamedArray, "seq"]]:  # type: ignore[name-defined]
    """Generate tokens using ``JitScheduler`` until either ``max_new_tokens`` have been
    produced *per sequence* or all sequences report finished."""

    def cond(state: tuple[GenState, jax.Array]):
        _gen_state, step = state
        finished = _gen_state.decode_state.is_finished(jnp.arange(_gen_state.decode_state.seq_id.size))
        return (step < max_rounds) & (_gen_state.sched.num_queued_tokens > 0) & (~jnp.all(finished))

    def body(state: tuple[GenState, jax.Array]) -> tuple[GenState, jax.Array]:
        gen_state: GenState
        gen_state, step = state

        # Pack the next chunk from the queue
        sched, packed_seq = gen_state.sched.pack_next_sequence(max_tokens_per_round)

        page_table, binfo = gen_state.page_table.allocate_for_seq(token_seq_ids=packed_seq.seq_ids)

        boundaries = packed_seq.boundary_indices(page_table.max_seqs)

        # jax.debug.print("Next tokens: {toks}, seq_ids: {seq_ids}, boundaries: {boundaries}",
        #                 toks=packed_seq.tokens, seq_ids=packed_seq.seq_ids, boundaries=boundaries)
        # Decode logits and sample new tokens
        # jax.debug.print("cache kvs: {cache_kvs}", cache_kvs=hax.nn.logsumexp(gen_state.cache.kv_pages, axis=("kv_head", "head_size")))
        logits, cache = model.decode(packed_seq.tokens, gen_state.cache, binfo, binfo.pos_ids)
        sample_key, key = jrandom.split(gen_state.prng_key)
        logits = logits["position", boundaries]
        cache = eqx.error_if(cache, hax.any(hax.isnan(cache.kv_pages)).scalar(), "New Cache contains NaNs")
        logits = eqx.error_if(logits, hax.any(hax.isnan(logits) & ~is_invalid(boundaries)).scalar(), "Logits contain NaNs")
        new_tokens, log_probs = sampler(logits, temps, key=sample_key)

        num_new_tokens = hax.sum(boundaries != INVALID).scalar().astype(jnp.int32)
        new_seq_ids = packed_seq.seq_ids["position", boundaries]
        # jax.debug.print("Sampled new tokens: {new_tokens}, num_new_tokens: {num_new_tokens}, seq_ids: {new_seq_ids}",
        #                 new_tokens=new_tokens, num_new_tokens=num_new_tokens, new_seq_ids=new_seq_ids)

        # Update scheduler with the freshly sampled tokens
        decode_state = gen_state.decode_state.update_tokens(new_seq_ids, new_tokens, log_probs, num_new_tokens)
        sched = sched.enqueue_tokens(new_tokens, new_seq_ids, num_new_tokens)

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(
            gen_state,
            sched=sched,
            page_table=page_table,
            cache=cache,
            decode_state=decode_state,
            prng_key=key
        )

        return new_gen_state, step + 1

    init_state = (gen_state, jnp.array(0, dtype=jnp.int32))
    final_gen_state, _ = jax.lax.while_loop(cond, body, init_state)

    decode_state, tokens, token_counts = final_gen_state.decode_state.extract_new_tokens()
    final_gen_state = dataclasses.replace(final_gen_state, decode_state=decode_state)

    return final_gen_state, tokens, token_counts


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

        temps = hax.full((), config.temperature, dtype=jnp.float32)

        total_prompt_tokens = sum(len(tokens) for tokens in prompt_ids)

        MAX_NEW_TOKENS = 32
        table = PageTable.init(64, len(prompt_ids), 8, 32)
        cache = haliax.named_jit(model.initial_cache)(table, dtype=config.trainer.mp.compute_dtype)
        sched = JitScheduler.init(256)
        decode_state = DecodeState.init(
            table.max_seqs,
            table.pages_per_seq,
            table.page_size,
            table.max_len_per_seq,
        )
        gen_state = GenState(
            sched=sched,
            cache=cache,
            page_table=table,
            decode_state=decode_state,
            prng_key=key
        )

        # -------------------------------- Scheduler-based generation --------------------------------
        for R in range(10):
            if R == 10:
                start_profiler("/tmp/gen-profile")
            elif R == 20:
                stop_profiler_and_maybe_wait(create_perfetto_link=False)
                levanter.current_tracker().log_artifact("/tmp/gen-profile", type="jax_profile")

            for i, toks in enumerate(prompt_ids):
                print(f"Prompt {i}: {toks}")

            time_in = time.time()
            outputs, gen_state, total_generated = _one_round(config, gen_state, model, prompt_ids, sampler, temps,
                                                  tokenizer, total_prompt_tokens, MAX_NEW_TOKENS)
            print(f"Round {R} took {time.time() - time_in:.2f} seconds, "
                  f"generated {total_generated} tokens in {len(outputs)} sequences.")

            # clear page table
            page_table = gen_state.page_table
            for seq_id in range(len(prompt_ids)):
                page_table = page_table.free_pages(seq_id)

            # Initialize GenState with all components
            gen_state = dataclasses.replace(
                gen_state,
                # in theory, we can just reuse the cache and not recreate it every time
                page_table=page_table,
                sched=sched.cleared(),
                decode_state=DecodeState.init(
                    page_table.max_seqs,
                    page_table.pages_per_seq,
                    page_table.page_size,
                    page_table.max_len_per_seq,
                ),
                prng_key=jrandom.PRNGKey(0)
            )
            del page_table


def _one_round(config, gen_state, model, prompt_ids, sampler, temps, tokenizer, total_prompt_tokens, MAX_NEW_TOKENS):
    time_in = time.time()
    finished = [False] * len(prompt_ids)
    outputs = [list(t) for t in prompt_ids]  # start with the prompts
    # do one prefill at a time, but we do continuous batching, so decode is happening all the while
    for tokens in prompt_ids:
        # enqueue the entire prompt into the scheduler
        page_table, seq_id = gen_state.page_table.assign_seq_id_to_seq()
        gen_state = dataclasses.replace(gen_state, page_table=page_table)

        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(tokens) + config.max_new_tokens, dtype=jnp.int32),
            stop_tokens=None,
            temperature=jnp.array(config.temperature, dtype=jnp.float32),
        )
        gen_state = dataclasses.replace(
            gen_state,
            decode_state=gen_state.decode_state.assign_seq(
                seq_id,
                seq_id,
                hax.full({"page": gen_state.page_table.pages_per_seq}, INVALID, dtype=jnp.int32),
                hax.named(np.asarray(tokens, dtype=jnp.int32), axis="position"),
                len(tokens),
                seq_params,
            ),
        )

        prompts_tokens_to_enqueue = np.asarray(tokens, dtype=jnp.int32)
        if len(tokens) > gen_state.sched.max_queued_tokens:
            raise ValueError(
                f"Prompt is too long ({len(tokens)} tokens), "
                f"max allowed is {MAX_NEW_TOKENS} tokens."
            )

        target_len = len(prompts_tokens_to_enqueue)
        if target_len > gen_state.sched.max_queued_tokens:
            print(
                f"Queue is full ({gen_state.sched.num_queued_tokens} tokens), running generation loop to free up space.")
        while target_len > gen_state.sched.empty_queue_space:
            # if the queue is too full, we run generation loop to free up space
            # TODO: would be better if we do partial/chunked prefill here, but hopefully this is rare
            gen_state, new_tokens, new_counts = run_generation_loop(
                gen_state,
                model,
                sampler,
                temps,
                # TODO: tune/configure
                64,
                32,
            )

            extract_outputs(new_tokens, new_counts, outputs, finished, total_prompt_tokens, config.max_new_tokens)

        this_tokens = hax.named(prompts_tokens_to_enqueue, axis="position")
        seq_ids = hax.full_like(this_tokens, seq_id, dtype=jnp.int32)
        new_sched = gen_state.sched.enqueue_tokens(this_tokens, seq_ids, prompts_tokens_to_enqueue.size)
        gen_state = dataclasses.replace(gen_state, sched=new_sched)
        del new_sched

        # do one macro-prefill round
        gen_state, new_tokens, new_counts = run_generation_loop(
            gen_state,
            model,
            sampler,
            temps,
            16,
            1,
        )

        extract_outputs(new_tokens, new_counts, outputs, finished, total_prompt_tokens, config.max_new_tokens)

    gen_state = jax.block_until_ready(gen_state)
    time_mid = time.time()
    print(f"Prefill took {time_mid - time_in:.2f} seconds, ")

    # finish chunked prefills and run autoregressive generation

    while not all(finished):
        time_gen_in = time.time()
        gen_state, new_tokens, new_counts = run_generation_loop(
            gen_state,
            model,
            sampler,
            temps,
            # TODO: tune/configure
            len(prompt_ids),
            64,
        )
        gen_state = jax.block_until_ready(gen_state)
        print(f"Generation loop iter took {time.time() - time_gen_in:.3f} seconds, ")
        time_gen_in = time.time()

        extract_outputs(new_tokens, new_counts, outputs, finished, total_prompt_tokens, config.max_new_tokens)
        print(f"Extracted outputs took {time.time() - time_gen_in:.3f} seconds, ")

    gen_state = jax.block_until_ready(gen_state)
    time_out = time.time()
    print(f"Gen loop took {time_out - time_mid:.2f} seconds, ")
    # Flatten, drop padding, and decode
    total_generated = sum(len(seq_outputs) for seq_outputs in outputs)
    total_generated -= sum(len(p) for p in prompt_ids)  # remove prompt tokens
    for seq_id, seq_outputs in enumerate(outputs):
        if finished[seq_id]:
            # remove padding tokens
            seq_outputs = [tok for tok in seq_outputs if tok != tokenizer.pad_token_id and tok != INVALID]
            outputs[seq_id] = seq_outputs
        else:
            print(f"Sequence {seq_id} did not finish, skipping decoding.")

        text = tokenizer.decode(seq_outputs, skip_special_tokens=True)
        print(f"Tokens for sequence {seq_id}: {seq_outputs}")
        print(f"Generated text for {seq_id}: {text}")
    return outputs, gen_state, total_generated


def extract_outputs(new_tokens, new_counts, outputs, finished, total_prompt_tokens, max_new_tokens):
    """
    drain generated tokens and append them to the outputs.

    MUTATES outputs and finished lists.
    """
    num_seqs = new_tokens.axis_size("seq")
    for seq_id in range(num_seqs):
        if finished[seq_id]:
            continue
        count = new_counts.array[seq_id]
        seq_tokens = new_tokens["seq", seq_id].array[:count]
        if seq_id >= len(outputs) or seq_id < 0:
            continue

        if len(outputs[seq_id]) + count >= max_new_tokens + total_prompt_tokens:
            # if we have enough tokens, mark this sequence as finished
            finished[seq_id] = True
            seq_tokens = seq_tokens[:max_new_tokens + total_prompt_tokens - len(outputs[seq_id])]

        outputs[seq_id].extend(seq_tokens)


if __name__ == "__main__":
    levanter.config.main(main)()
