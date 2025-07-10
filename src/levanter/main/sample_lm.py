import logging
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.layers.attention import KvPageState, PageTable
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

    prompt: str = "Hello"
    max_new_tokens: int = 20
    temperature: float = 0.8


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


def main(config: SampleLmConfig):
    levanter.initialize(config)
    tokenizer = load_tokenizer(config.tokenizer)

    vocab_size = len(tokenizer)
    Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)

    key = jrandom.PRNGKey(0)

    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.parameter_axis_mapping):
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

        page_table, binfo = page_table.allocate_for_seqs(
            updated_seqs=hax.named([seq_id], "seq"),
            new_counts=hax.named([len(prompt_ids)], "seq"),
            tokens=hax.named([seq_id] * len(prompt_ids), prompt_axis),
        )
        state = KvPageState.from_batch(binfo, cache)
        pos_ids = hax.arange(prompt_axis, dtype=jnp.int32)
        _, state = model.decode(prompt_tokens, state, pos_ids)

        generated = list(prompt_ids)
        temps = hax.full(Axis("batch", 1), config.temperature, dtype=jnp.float32)

        for i in range(config.max_new_tokens):
            page_table, binfo = page_table.allocate_for_seqs(
                updated_seqs=hax.named([seq_id], "seq"),
                new_counts=hax.named([1], "seq"),
                tokens=hax.named([seq_id], Axis("position", 1)),
            )
            state = KvPageState.from_batch(binfo, state.cache)
            pos_id = hax.arange(Axis("position", 1), start=len(generated))
            logits, state = model.decode(
                hax.NamedArray(jnp.array([generated[-1]], dtype=jnp.int32), axes=(Axis("position", 1),)),
                state,
                pos_id,
            )
            logits = logits["position", 0]
            tok, _ = sampler(logits, temps, key=jrandom.PRNGKey(i + 1))
            next_token = int(tok.array)
            generated.append(next_token)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(text)


if __name__ == "__main__":
    levanter.config.main(main)()
