import logging
from dataclasses import dataclass, field
from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.layers.page_table import PageTable
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from .sample_lm import do_prefill
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.inference.jit_scheduler import JitScheduler

logger = logging.getLogger(__name__)


@dataclass
class BatchSampleConfig:
    """Configuration for batch text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    prompts: List[str] = field(default_factory=list)
    max_new_tokens: int = 100
    temperature: float = 1e-4
    max_tokens: int = 4096


def _load_model(config: BatchSampleConfig, Vocab: Axis, *, key) -> LmHeadModel:
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
        converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=load_tokenizer(config.tokenizer))
        model = converter.load_pretrained(config.model.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype)
        return model


def _prefill_one(prompt_tokens, seq_id, sched, page_table, cache, model, sampler, temps, key):
    seq_ids = hax.full_like(prompt_tokens, seq_id, dtype=jnp.int32)
    sched = sched.enqueue_tokens(prompt_tokens, seq_ids, prompt_tokens.size)
    sched, chunk_tokens, chunk_seq_ids = sched.pack_next_sequence(prompt_tokens.size)
    next_tok, page_table, cache = do_prefill(
        model,
        cache,
        page_table,
        chunk_tokens,
        chunk_seq_ids,
        sampler,
        temps,
        key,
    )
    sched = sched.update_after_sampling(
        new_tokens=hax.named(jnp.array([next_tok.item()], dtype=jnp.int32), axis="position"),
        new_token_seq_ids=hax.named(jnp.array([chunk_seq_ids.array[0]], dtype=jnp.int32), axis="position"),
        num_new_tokens=1,
    )
    return sched, page_table, cache


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
    """Generate tokens using ``JitScheduler`` until either ``max_new_tokens`` have been produced per sequence."""

    def cond(state):
        _sched, *_ , step = state
        return (step < max_new_tokens) & (~jnp.all(_sched.finished.array))

    def body(state):
        sched: JitScheduler
        sched, page_table, cache, key, step = state

        sched, chunk_tokens, chunk_seq_ids = sched.pack_next_sequence(max_tokens_per_round)
        page_table, binfo = page_table.allocate_for_seq(token_seq_ids=chunk_seq_ids)
        logits, cache = model.decode(chunk_tokens, cache, binfo, binfo.pos_ids)
        sample_key, key = jrandom.split(key)
        logits = logits["position", binfo.last_token_idx]
        new_tokens, _ = sampler(logits, temps, key=sample_key)
        num_new_tokens = hax.sum(binfo.last_token_idx != -1).scalar()
        sched = sched.update_after_sampling(
            new_tokens=new_tokens,
            new_token_seq_ids=chunk_seq_ids,
            num_new_tokens=num_new_tokens,
        )
        return sched, page_table, cache, key, step + 1

    init_state = (sched, page_table, cache, key, jnp.array(0, dtype=jnp.int32))
    sched, page_table, cache, key, _ = jax.lax.while_loop(cond, body, init_state)
    return sched, cache, page_table, key


def main(config: BatchSampleConfig):
    levanter.initialize(config)

    if not config.prompts:
        raise ValueError("No prompts provided")

    tok_string: str | None = config.tokenizer
    if config.tokenizer is None:
        if config.hf_checkpoint is not None:
            tok_string = config.hf_checkpoint.model_name_or_path

    if tok_string is None:
        raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")

    tokenizer = load_tokenizer(tok_string)

    key = jrandom.PRNGKey(0)

    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        model = _load_model(config, Vocab, key=key)
        assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

        sampler = Sampler(Vocab)

        prompts_tokens = []
        for text in config.prompts:
            p_ids = tokenizer.encode(text, add_special_tokens=False)
            p_axis = Axis("position", len(p_ids))
            prompts_tokens.append(hax.NamedArray(jnp.array(p_ids, dtype=jnp.int32), axes=(p_axis,)))

        num_prompts = len(prompts_tokens)
        page_table = PageTable.init(
            max_pages=100 * num_prompts,
            max_seqs=num_prompts,
            page_size=16,
            max_pages_per_seq=32,
        )
        cache = eqx.filter_jit(model.initial_cache)(page_table, dtype=jnp.bfloat16)
        cache = hax.auto_sharded(cache)

        temps = hax.full((), config.temperature, dtype=jnp.float32)

        sched = JitScheduler.init(
            max_tokens=config.max_tokens,
            max_seqs=num_prompts,
            key=key,
        )

        seq_ids = []
        for pt in prompts_tokens:
            page_table, seq_id = page_table.assign_seq_id_to_seq()
            seq_ids.append(seq_id)
            sched, page_table, cache = _prefill_one(pt, seq_id, sched, page_table, cache, model, sampler, temps, key)

        sched, cache, page_table, key = run_generation_loop(
            sched,
            page_table,
            cache,
            model,
            sampler,
            temps,
            key,
            1,
            config.max_new_tokens,
        )

        results = []
        for pt, seq_id in zip(prompts_tokens, seq_ids):
            out_ids = hax.named(jnp.array([seq_id], dtype=jnp.int32), axis="seq")
            sched, output_matrix = sched.extract_generated_tokens(out_ids, max_tokens=config.max_new_tokens)
            generated_token_ids = [int(t) for t in pt.array]
            generated_token_ids.extend([int(tok) for tok in output_matrix.array[0] if tok != -1])
            text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            results.append(text)

        for r in results:
            print(f"Generated text: {r}")

        return results


if __name__ == "__main__":
    levanter.config.main(main)()
