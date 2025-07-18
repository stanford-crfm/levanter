import shutil

import jax
import time

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
        model = converter.load_pretrained(config.model.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype)
        return model


# @haliax.named_jit(donate_args=(False, True, True, False, False, True))
# def do_prefill(model, cache, page_table, tokens, sampler, temps, key):
#     """Prefill ``tokens`` and sample the next token."""
#     pos_axis = tokens.axes[0]
#     # TODO: actual batch
#     page_table, binfo = page_table.allocate_for_seq(
#         token_seq_ids=hax.zeros_like(tokens, dtype=jnp.int32)
#     )
#
#     pos_ids = hax.arange(pos_axis, dtype=jnp.int32)
#     logits, cache = model.decode(tokens, cache, binfo, pos_ids)
#     next_tok, _ = sampler(logits["position", -1], temps, key=key)
#     return next_tok, page_table, cache


@haliax.named_jit(donate_args=(False, True, True, False, False, True))
def do_prefill(model, cache, page_table: PageTable, tokens, seq_ids, sampler, temps, key):
    """Prefill ``tokens`` and sample the next token."""
    page_table, binfo = page_table.allocate_for_seq(token_seq_ids=seq_ids)
    # TODO: this is potentially a bit wasteful since we don't need to compute logits for all but the last token

    logits, cache = model.decode(tokens, cache, binfo, binfo.pos_ids)
    next_tok, _ = sampler(logits["position", -1], temps, key=key)
    return next_tok, page_table, cache



def tree_byte_size(tree):
    """Calculate the total byte size of a JAX tree."""

    # TODO: take into account sharding
    def _leaf_size(x):
        if is_jax_array_like(x):
            return x.nbytes
        return 0

    return sum(_leaf_size(x) for x in jax.tree.leaves(tree))


@eqx.filter_jit
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
        _sched, *_ , step = state
        return (step < max_new_tokens) & (~jnp.all(_sched.finished.array))

    def body(state):
        sched: JitScheduler
        sched, page_table, cache, key, step = state

        # Pack the next chunk from the queue
        sched, chunk_tokens, chunk_seq_ids = sched.pack_next_sequence(max_tokens_per_round)


        # Allocate cache pages for this chunk
        page_table, binfo = page_table.allocate_for_seq(token_seq_ids=chunk_seq_ids)

        jax.debug.print("Running generation step {step} with chunk_tokens={chunk_tokens}, chunk_seq_ids={chunk_seq_ids}, binfo={binfo}",
                        step=step, chunk_tokens=chunk_tokens, chunk_seq_ids=chunk_seq_ids, binfo=binfo)

        # Decode logits and sample new tokens
        logits, cache = model.decode(chunk_tokens, cache, binfo, binfo.pos_ids)
        sample_key, key = jrandom.split(key)
        new_tokens, _ = sampler(logits, temps, key=sample_key)

        # Update scheduler with the freshly sampled tokens
        sched = sched.update_after_sampling(
            new_tokens=new_tokens,
            new_token_seq_ids=chunk_seq_ids,
            num_new_tokens=max_tokens_per_round,
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
        prompt_axis = Axis("position", len(prompt_ids))
        prompt_tokens = hax.NamedArray(jnp.array(prompt_ids, dtype=jnp.int32), axes=(prompt_axis,))

        page_table = PageTable.init(
            max_pages=100,
            max_seqs=16,
            page_size=16,
            max_pages_per_seq=32,
        )
        cache = eqx.filter_jit(model.initial_cache)(page_table, dtype=jnp.bfloat16)
        cache = hax.auto_sharded(cache)

        temps = hax.full((), config.temperature, dtype=jnp.float32)

        # ----------------------- Scheduler init -----------------------
        MAX_TOKENS = 32    # per‐round chunk size
        MAX_SEQS = 16      # hot‐set size (matches page_table.max_seqs)
        sched = JitScheduler.init(
            max_tokens=MAX_TOKENS,
            max_seqs=MAX_SEQS,
            key=key,
        )
        # --------------------------------------------------------------

        model_size = tree_byte_size(model)

        size = tree_byte_size(
            (model, cache, page_table, prompt_tokens, sampler, temps)
        )

        print(f"Arg sizes for pre_fill {size / 1024 ** 2:.2f} MB:\n"
              f"  model: {model_size / 1024 ** 2:.2f} MB\n"
              f"  cache: {tree_byte_size(cache) / 1024 ** 2:.2f} MB\n"
              f"  page_table: {tree_byte_size(page_table) / 1024 ** 2:.2f} MB\n"
              f"  prompt_tokens: {tree_byte_size(prompt_tokens) / 1024 ** 2:.2f} MB\n"
              f"  sampler: {tree_byte_size(sampler) / 1024 ** 2:.2f} MB\n"
              f"  temps: {tree_byte_size(temps) / 1024 ** 2:.2f} MB")

        # -------------------------------- Scheduler-based generation --------------------------------
        prng_key = jrandom.PRNGKey(0)
        page_table = page_table.free_pages(0)
        page_table, seq_id = page_table.assign_seq_id_to_seq()

        # enqueue the entire prompt into the scheduler
        seq_ids = hax.full_like(prompt_tokens, seq_id, dtype=jnp.int32)
        sched = sched.enqueue_tokens(prompt_tokens, seq_ids, prompt_tokens.size)
        jax.debug.print("{queue}", queue=sched.queued_tokens)

        # do one macro-prefill round
        sched, chunk_tokens, chunk_seq_ids = sched.pack_next_sequence(MAX_TOKENS)
        next_tok, page_table, cache = do_prefill(
            model, cache, page_table,
            chunk_tokens, chunk_seq_ids,
            sampler, temps, prng_key
        )
        sched = sched.update_after_sampling(
            new_tokens=hax.named(jnp.array([next_tok.array.item()], dtype=jnp.int32), axis="position"),
            new_token_seq_ids=hax.named(jnp.array([chunk_seq_ids.array[0]], dtype=jnp.int32), axis="position"),
            num_new_tokens=1,
        )

        # run the fully JIT-compiled generation loop
        sched, cache, page_table, prng_key = run_generation_loop(
            sched,
            page_table,
            cache,
            model,
            sampler,
            temps,
            prng_key,
            MAX_TOKENS,
            config.max_new_tokens,
        )

        # extract up to `max_new_tokens` tokens for this sequence
        out_ids = hax.named(jnp.array([seq_id], dtype=jnp.int32), axis="seq")
        sched, output_matrix = sched.extract_generated_tokens(out_ids, max_tokens=config.max_new_tokens)

        # Flatten, drop padding, and decode
        generated_token_ids = [int(t) for t in prompt_ids]
        generated_token_ids.extend([int(tok) for tok in output_matrix.array[0] if tok != -1])
        text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"Generated text: {text}")
        # -------------------------------------------------------------------------------------------

        return text



# @hax.named_jit(donate_args=(False, True, True, False, False, False, True))
# @equinox.debug.assert_max_traces(max_traces=4)
def do_generate(model, cache, binfo, prev_token, sampler, pos_ids, temps, prng_key):
    prev_token = hax.named(prev_token, "position")

    logits, cache = model.decode(prev_token, cache, binfo, pos_ids)
    logits = logits["position", 0]
    tok, _ = sampler(logits, temps, key=prng_key)
    return tok, cache


@haliax.named_jit(donate_args=(False, False, True, True, False, False, False))
def do_generate_n_times(n, model, cache, page_table, prev_token, sampler, temps, prng_key):
    """Generate `n` tokens starting from `prev_token`."""
    generated_tokens = jnp.full(n, -1, dtype=jnp.int32)

    def do_block(i, gen_tokens, prev_token, page_table: PageTable, cache, prng_key):
        page_table, binfo = page_table.allocate_for_seq(token_seq_ids=hax.zeros({"position": 1}, dtype=jnp.int32))
        this_key, prng_key = jrandom.split(prng_key, 2)
        pos_id = page_table.pos_ids_from_seq_ids(hax.zeros({"position": 1}, dtype=jnp.int32))
        # jax.debug.print("Generating token {i} with prev_token={prev_token}, pos_id={pos_id}, pt_lens={pt_lens}", i=i, prev_token=prev_token, pos_id=pos_id, pt_lens=page_table.seq_lens)
        # tok, page_table, cache = do_generate(model, cache, page_table, prev_token, sampler, pos_id, temps, this_key)
        tok, cache = do_generate(model, cache, binfo, prev_token, sampler, pos_id, temps, this_key)
        gen_tokens = gen_tokens.at[i].set(tok.scalar())
        return gen_tokens, tok.scalar().reshape((1,)), page_table, cache, prng_key

    gen_tokens, last_tok, page_table, cache, _ = jax.lax.fori_loop(0, n, lambda i, args: do_block(i, *args),
                                                                   (generated_tokens, prev_token, page_table, cache,
                                                                    prng_key),
                                                                   unroll=4)
    return gen_tokens, page_table, cache


if __name__ == "__main__":
    levanter.config.main(main)()
