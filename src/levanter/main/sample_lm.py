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
from haliax import Axis, NamedArray
from haliax.partitioning import round_axis_for_partitioning

import levanter
from haliax.jax_utils import is_jax_array_like

# from levanter.callbacks import start_profiler, stop_profiler_and_maybe_wait
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.page_table import PageTable
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.inference.jit_scheduler import DecodeState, SeqDecodingParams, TokenQueue
from levanter.inference.utils import INVALID, is_valid
from levanter.layers.attention import KvPageCache

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Request:
    """A request for generation of a single sequence."""
    prompt_tokens: list[int]
    request_id: int
    decode_params: SeqDecodingParams

SEQ_LENGTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096]


class GenState(eqx.Module):
    """
    Plain Old Data type for generation state.
    Contains all the components needed for language model generation.
    """
    cache: KvPageCache
    page_table: PageTable
    decode_state: DecodeState


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
    stop_sequence: str | None = "\n"
    "Stop sequences. Currently only does whole token sequences."
    max_new_tokens: int = 192
    temperature: float = 0.7
    seed: int = 2


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
    max_tokens_per_round: int,
    max_rounds: int,
) -> GenState:
    """Generate tokens until all sequences are finished or max rounds is reached."""

    def cond(state: tuple[GenState, jax.Array, jax.Array]):
        _gen_state, finished, step = state
        return (step < max_rounds) & (_gen_state.decode_state.num_queued_tokens > 0) & (~jnp.all(finished))

    def body(state: tuple[GenState, jax.Array, jax.Array]) -> tuple[GenState, jax.Array, jax.Array]:
        gen_state, has_finished, step = state

        # Pack the next chunk from the queue via DecodeState
        decode_state, packed_seq = gen_state.decode_state.pack_next_sequence(max_tokens_per_round)

        page_table, binfo = gen_state.page_table.allocate_for_seq(token_seq_ids=packed_seq.seq_ids)

        boundaries = packed_seq.boundary_indices(min(page_table.max_seqs, max_tokens_per_round))

        # Decode logits and sample new tokens
        logits, cache = model.decode(packed_seq.tokens, gen_state.cache, binfo, binfo.pos_ids)
        logits = logits["position", boundaries]
        # cache = eqx.error_if(cache, hax.any(hax.isnan(cache.kv_pages)).scalar(), "New Cache contains NaNs")
        # logits = eqx.error_if(logits, hax.any(hax.isnan(logits) & ~is_invalid(boundaries)).scalar(), "Logits contain NaNs")

        num_new_tokens = hax.sum(boundaries != INVALID).scalar().astype(jnp.int32)
        new_seq_ids = packed_seq.seq_ids["position", boundaries]
        new_pos_ids = binfo.pos_ids["position", boundaries]
        prng_keys = decode_state.prng_keys_for(new_seq_ids, new_pos_ids)

        temps = decode_state.temperature["seq", new_seq_ids]

        new_tokens, log_probs = hax.vmap(sampler, "position")(logits, temps, key=prng_keys)

        # Update decode state with the freshly sampled tokens (also enqueues them)
        decode_state = decode_state.update_tokens(new_tokens, new_seq_ids, log_probs, num_new_tokens)
        new_finished = decode_state.is_finished(jnp.arange(decode_state.max_seqs))
        has_finished = has_finished | new_finished

        # purge any finished sequencse
        finished_sequences = jnp.nonzero(new_finished, size=gen_state.page_table.max_seqs, fill_value=INVALID)[0]
        finished_sequences = hax.named(finished_sequences, axis="seq")
        decode_state = decode_state.purge_queue_of_seq(finished_sequences)

        # Update the gen_state with all the new components
        new_gen_state = dataclasses.replace(
            gen_state,
            page_table=page_table,
            cache=cache,
            decode_state=decode_state,
        )

        return new_gen_state, has_finished, step + 1

    has_finished = gen_state.decode_state.is_finished(jnp.arange(gen_state.decode_state.max_seqs))
    init_state = (gen_state, has_finished, jnp.array(0, dtype=jnp.int32))
    final_gen_state, has_finished, _ = jax.lax.while_loop(cond, body, init_state)

    return final_gen_state


@haliax.named_jit(donate_args=(True, False, False, False, False))
def run_prefill_loop(
    gen_state: GenState,
    model,
    sampler,
    flat_tokens,
    flat_seq_ids,
    max_tokens_per_round: int,
) -> GenState:
    """Run prefill using a fresh, local token queue. Newly sampled tokens are enqueued to the main decode queue via update_tokens."""

    prefill_queue = TokenQueue.init(flat_tokens.axis_size("position"))
    prefill_queue = prefill_queue.enqueue_tokens(flat_tokens, flat_seq_ids, flat_tokens.axis_size("position"))

    def cond(state: tuple[GenState, jax.Array, TokenQueue]):
        _gen_state, step, local_q = state
        return (local_q.num_queued_tokens > 0) & (step < 10_000)

    def body(state: tuple[GenState, jax.Array, TokenQueue]):
        gen_state, step, local_q = state

        local_q, packed_seq = local_q.pack_next_sequence(max_tokens_per_round)

        page_table, binfo = gen_state.page_table.allocate_for_seq(token_seq_ids=packed_seq.seq_ids)

        boundaries = packed_seq.boundary_indices(min(page_table.max_seqs, max_tokens_per_round))

        logits, cache = model.decode(packed_seq.tokens, gen_state.cache, binfo, binfo.pos_ids)
        logits = logits["position", boundaries]

        num_new_tokens = hax.sum(boundaries != INVALID).scalar().astype(jnp.int32)
        new_seq_ids = packed_seq.seq_ids["position", boundaries]
        new_pos_ids = binfo.pos_ids["position", boundaries]
        prng_keys = gen_state.decode_state.prng_keys_for(new_seq_ids, new_pos_ids)
        temps = gen_state.decode_state.temperature["seq", new_seq_ids]

        new_tokens, log_probs = hax.vmap(sampler, "position")(logits, temps, key=prng_keys)

        # Update decode_state (also enqueues into the main decode queue)
        decode_state = gen_state.decode_state.update_tokens(new_tokens, new_seq_ids, log_probs, num_new_tokens)

        # Purge any finished sequences from the local prefill queue
        new_finished = decode_state.is_finished(jnp.arange(decode_state.max_seqs))
        finished_sequences = jnp.nonzero(new_finished, size=gen_state.page_table.max_seqs, fill_value=INVALID)[0]
        finished_sequences = hax.named(finished_sequences, axis="seq")
        local_q = local_q.purge_queue_of_seq(finished_sequences)

        gen_state = dataclasses.replace(gen_state, page_table=page_table, cache=cache, decode_state=decode_state)
        return gen_state, step + 1, local_q

    init_state = (gen_state, jnp.array(0, dtype=jnp.int32), prefill_queue)
    gen_state, _, _ = jax.lax.while_loop(cond, body, init_state)
    return gen_state


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

        sampler = Sampler(Vocab)

        prompts = config.prompts

        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

        table = PageTable.init(64, len(prompt_ids), 8, 32)
        cache = haliax.named_jit(model.initial_cache)(table, dtype=config.trainer.mp.compute_dtype)
        initial_decode_state = DecodeState.init(
            table.max_seqs,
            table.pages_per_seq,
            table.page_size,
            table.max_len_per_seq,
            max_stop_seqs=1,
            max_queued_tokens=32,
        )
        gen_state = GenState(
            cache=cache,
            page_table=table,
            decode_state=initial_decode_state,
        )

        # -------------------------------- Scheduler-based generation --------------------------------

        stop_sequence = config.stop_sequence
        if stop_sequence is not None:
            stop_ids = tokenizer(stop_sequence, add_special_tokens=False)["input_ids"]
            if len(stop_ids) == 0:
                raise ValueError("Stop sequence must be non-empty")
            stop_ids = hax.named(np.asarray(stop_ids, dtype=np.int32), axis="position")
        else:
            stop_ids = None

        for R in range(10):
            for i, toks in enumerate(prompt_ids):
                print(f"Prompt {i}: {toks}")

            time_in = time.time()
            outputs, gen_state, total_generated = _one_round(
                config, gen_state, model, prompt_ids, sampler, tokenizer, stop_ids, key
            )
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
                decode_state=initial_decode_state,
            )
            del page_table


def _one_round(config, gen_state, model, prompt_ids, sampler, tokenizer, stop_ids: NamedArray | None, key):
    time_in = time.time()
    finished = [False] * len(prompt_ids)
    outputs = [list(t) for t in prompt_ids]  # start with the prompts

    requests: list[Request] = []
    if stop_ids is not None:
        stop_ids = stop_ids.broadcast_axis({"stop_seq": 1})

    for req_id, toks in enumerate(prompt_ids):
        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(len(toks) + config.max_new_tokens, dtype=jnp.int32),
            stop_tokens=stop_ids,
            temperature=jnp.array(config.temperature, dtype=jnp.float32),
            key=jax.random.fold_in(key, req_id),
        )
        requests.append(Request(prompt_tokens=toks, request_id=req_id, decode_params=seq_params))

    gen_state = prefill_prompts(gen_state, model, sampler, requests)

    extract_outputs(gen_state.decode_state, outputs, finished)

    gen_state = jax.block_until_ready(gen_state)
    time_mid = time.time()
    print(f"Prefill took {time_mid - time_in:.2f} seconds, ")

    # finish chunked prefills and run autoregressive generation

    while not all(finished):
        time_gen_in = time.time()
        gen_state = run_generation_loop(
            gen_state,
            model,
            sampler,
            # TODO: tune/configure
            len(requests),
            8,
        )
        total_gen_loop = time.time() - time_gen_in
        gen_state = jax.block_until_ready(gen_state)
        time_gen_in = time.time()

        new_tokens = extract_outputs(gen_state.decode_state, outputs, finished)
        print(f"Extracted outputs took {time.time() - time_gen_in:.3f} seconds")
        tps = new_tokens / total_gen_loop
        print(f"Generation loop iter took {total_gen_loop :.3f} seconds, {tps:.2f} tokens/sec, ")

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
        print(f"Tokens for sequence {seq_id} (len: {len(seq_outputs)}: {seq_outputs}")

        print(f"Generated text for {seq_id}: {text}")
        # zip tokens with their individial text
        tokens = [f"{tok} ({tokenizer.decode([tok], skip_special_tokens=True)})" for tok in seq_outputs]
        print(f"Tokens with text for sequence {seq_id}: {tokens}")

    return outputs, gen_state, total_generated


def pad_to_standard_length(tokens: np.ndarray, allowed_lengths: list[int], pad_token_id: int) -> np.ndarray:
    """Pad the token array to the nearest allowed length using the pad_token_id."""
    current_length = tokens.shape[0]
    target_length = min((length for length in allowed_lengths if length >= current_length), default=None)

    if target_length is None:
        raise ValueError(f"Current length {current_length} exceeds all allowed lengths {allowed_lengths}")

    padding_length = target_length - current_length
    if padding_length > 0:
        padding = np.full((padding_length,), pad_token_id, dtype=tokens.dtype)
        tokens = np.concatenate([tokens, padding], axis=0)

    return tokens


def prefill_prompts(
    gen_state: GenState,
    model,
    sampler,
    prompts: list[Request],
) -> tuple[GenState, dict[int, list[int]]]:
    """Assign seq ids, set params, and run prefill via a fresh token queue for all prompts."""
    all_tokens = []
    all_seq_ids = []

    for request in prompts:
        seq_tokens = request.prompt_tokens
        seq_params = request.decode_params

        page_table, seq_id = gen_state.page_table.assign_seq_id_to_seq()
        seq_id = int(seq_id)

        if not is_valid(seq_id):
            raise RuntimeError("Ran out of sequence IDs in the page table during prefill.")

        gen_state = dataclasses.replace(gen_state, page_table=page_table)

        tokens_array = np.asarray(seq_tokens, dtype=jnp.int32)
        # build allowed lengths = multiples of page size up to pages_per_seq
        page_size = gen_state.page_table.page_size
        pages_per_seq = gen_state.page_table.pages_per_seq
        allowed_lengths = [page_size * k for k in range(1, pages_per_seq + 1)]
        padded = pad_to_standard_length(tokens_array, allowed_lengths, INVALID)
        gen_state = dataclasses.replace(
            gen_state,
            decode_state=gen_state.decode_state.assign_seq(
                local_seq_id=seq_id,
                global_seq_id=request.request_id,
                tokens=hax.named(padded, axis="position"),
                prefix_len=len(seq_tokens),
                kv_pages=None,
                seq_params=seq_params,
            ),
        )

        sids = np.full_like(tokens_array, seq_id, dtype=jnp.int32)
        all_tokens.append(tokens_array)
        all_seq_ids.append(sids)

    if len(all_tokens) > 0:
        flat_tokens = jnp.concatenate(all_tokens)
        flat_seq_ids = jnp.concatenate(all_seq_ids)
        flat_tokens = hax.named(flat_tokens, axis="position")
        flat_seq_ids = hax.named(flat_seq_ids, axis="position")
        gen_state = run_prefill_loop(gen_state, model, sampler, flat_tokens, flat_seq_ids, max_tokens_per_round=64)

    return gen_state


def extract_outputs(decode_state: DecodeState, outputs, finished):
    """
    drain generated tokens and append them to the outputs.

    MUTATES outputs and finished lists.
    """
    total_this_time = 0
    num_seqs = decode_state.max_seqs
    tokens = jax.device_get(decode_state.tokens.array)
    num_tokens = jax.device_get(decode_state.num_tokens.array)
    seq_ids = jax.device_get(decode_state.seq_id.array)
    this_finished = jax.device_get(decode_state.is_finished(jnp.arange(num_seqs)))

    for local_seq in range(num_seqs):
        global_id = int(seq_ids[local_seq])
        if global_id < 0 or global_id == INVALID:
            continue

        current_num_tokens = len(outputs[global_id])
        new_num_tokens = int(num_tokens[local_seq])
        count_to_extract = new_num_tokens - current_num_tokens

        if finished[global_id]:
            continue

        total_this_time += max(0, count_to_extract)

        if bool(this_finished[local_seq]):
            finished[global_id] = True

        if count_to_extract <= 0:
            continue

        seq_tokens = tokens[local_seq, current_num_tokens:new_num_tokens]
        outputs[global_id].extend(int(x) for x in seq_tokens)

    return total_this_time


if __name__ == "__main__":
    levanter.config.main(main)()
