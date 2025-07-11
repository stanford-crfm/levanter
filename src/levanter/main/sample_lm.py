import equinox.debug
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
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.layers.page_table import PageTable
from levanter.layers.sampler import Sampler
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

    tokenizer: str = "meta-llama/Llama-2-7b-hf"

    prompt: str = "Four score and seven years ago"
    max_new_tokens: int = 20
    temperature: float = 0.2


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
        return model
    else:
        assert hasattr(config.model, "hf_checkpoint_converter"), "model config lacks HF loader"
        converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=config.hf_checkpoint, tokenizer=load_tokenizer(config.tokenizer))
        model = converter.load_pretrained(config.model.model_type, ref=config.hf_checkpoint, dtype=mp.compute_dtype)
        return model



@haliax.named_jit(donate_args=(False, True, True, False, False, False, True))
def do_prefill(model, cache, page_table, tokens, sampler, seq_id, temps, key):
    """Prefill ``tokens`` and sample the next token."""
    pos_axis = tokens.axes[0]
    page_table, binfo = page_table.allocate_for_seqs(
        updated_seqs=seq_id,
        new_counts=hax.named([pos_axis.size], "seq"),
        tokens=hax.named([seq_id.array[0]] * pos_axis.size, pos_axis),
    )

    pos_ids = hax.arange(pos_axis, dtype=jnp.int32)
    logits, cache = model.decode(tokens, cache, binfo, pos_ids)
    next_tok, _ = sampler(logits["position", -1], temps, key=key)
    return next_tok, page_table, cache


@haliax.named_jit(donate_args=(False, False, False, True, True, True, True))
def jit_decode_fn(model, sampler, temps, tokens, state, pos_id, key):
    k1, k2 = hax.split(key, 2)
    logits, state = model.decode(tokens, state, pos_id, key=k1)
    token, log_prob = sampler(logits["position", 0], temps, key=key)
    return token, log_prob, state


def main(config: SampleLmConfig):
    levanter.initialize(config)
    tokenizer = load_tokenizer(config.tokenizer)

    vocab_size = len(tokenizer)
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)

    key = jrandom.PRNGKey(0)

    # NB: we use the compute_axis_mapping b/c we're doing inference
    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.compute_axis_mapping):
        model = _load_model(config, Vocab, key=key)
        assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

        sampler = Sampler(Vocab)

        prompt_ids = tokenizer.encode(config.prompt, add_special_tokens=False)
        prompt_axis = Axis("position", len(prompt_ids))
        prompt_tokens = hax.NamedArray(jnp.array(prompt_ids, dtype=jnp.int32), axes=(prompt_axis,))

        page_table = PageTable.init(
            max_pages=1,
            max_seqs=1,
            page_size=len(prompt_ids) + config.max_new_tokens,
            max_pages_per_seq=1,
        )
        page_table, seq_id = page_table.assign_seq_id_to_seq()
        cache = model.initial_cache(page_table, dtype=jnp.float32)

        seq_named = hax.named([seq_id], "seq")
        temps = hax.full((), config.temperature, dtype=jnp.float32)

        prng_key = jrandom.PRNGKey(0)
        tok, page_table, cache = do_prefill(
            model, cache, page_table, prompt_tokens, sampler, seq_named, temps, prng_key
        )

        generated = list(prompt_ids) + [int(tok.array)]

        token_times = []

        for i in range(1, config.max_new_tokens):
            time_in = time.time()
            prng_key = jrandom.PRNGKey(i)
            prev_token = jnp.array([generated[-1]], dtype=jnp.int32)
            start = jnp.array(len(generated), dtype=jnp.int32)

            tok, page_table, cache, = do_generate(model, cache, page_table, prev_token, sampler, seq_named, start, temps, prng_key)
            next_token = int(tok.array)
            time_out = time.time()
            token_times.append(time_out - time_in)
            generated.append(next_token)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(text)
        print(f"Generated {len(generated) - len(prompt_ids)} tokens in {sum(token_times):.2f} seconds")
        print(token_times)


@hax.named_jit(donate_args=(False, True, True, False, False, False, True))
@equinox.debug.assert_max_traces(max_traces=4)
def do_generate(model, cache, page_table, prev_token, sampler, seq_id, start, temps, prng_key):
    prev_token = hax.named(prev_token, "position")

    page_table, binfo = page_table.allocate_for_seqs(
        updated_seqs=seq_id,
        new_counts=hax.named([1], "seq"),
        tokens=seq_id.rename({"seq": "position"})
    )
    pos_id = hax.arange(Axis("position", 1), start=start)
    logits, cache = model.decode(
        prev_token,
        cache,
        binfo,
        pos_id,
    )
    logits = logits["position", 0]
    tok, _ = sampler(logits, temps, key=prng_key)
    return tok, page_table, cache


if __name__ == "__main__":
    levanter.config.main(main)()
