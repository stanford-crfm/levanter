from __future__ import annotations

from typing import List, cast

import jax.numpy as jnp
import jax.random as jrandom
from transformers import AutoTokenizer

import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

from levanter.compat.hf_checkpoints import RepoRef
from levanter.layers.page_table import PageTable
from levanter.layers.sampler import Sampler
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.trainer import TrainerConfig

from .sampling_params import SamplingParams
from .sequence import Sequence, SequenceStatus
from .scheduler import Scheduler
from levanter.main.sample_lm import do_prefill, do_generate


PREFERRED_SIZES = (1, 4, 16, 32, 64, 256, 1024, 4096, 165336)
MAX_SEQS = 32


def _round_preferred(n: int) -> int:
    for s in PREFERRED_SIZES:
        if n <= s:
            return s
    return PREFERRED_SIZES[-1]


class LLMEngine:
    """Simplified engine for autoregressive generation."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.trainer_cfg = TrainerConfig()
        self.Vocab = round_axis_for_partitioning(
            Axis("vocab", len(self.tokenizer)), self.trainer_cfg.compute_axis_mapping
        )
        converter = LlamaConfig().hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=RepoRef(model_path), tokenizer=self.tokenizer)
        self.model = cast(
            LlamaLMHeadModel,
            converter.load_pretrained(
                LlamaLMHeadModel,
                ref=RepoRef(model_path),
                dtype=self.trainer_cfg.mp.compute_dtype,
            ),
        )
        self.sampler = Sampler(self.Vocab)
        self.eos = self.tokenizer.eos_token_id or -1

    # ------------------------------------------------------------------
    def add_request(self, prompt: str | List[int], sampling_params: SamplingParams, scheduler: Scheduler) -> None:
        prompt_ids = (
            self.tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str)
            else prompt
        )
        seq = Sequence(list(prompt_ids), sampling_params)
        scheduler.add(seq)

    def _prefill(self, seq: Sequence, cache, page_table):
        real_len = len(seq.prompt_token_ids)
        padded_len = _round_preferred(real_len)
        pos_axis = Axis("position", padded_len)
        padded_tokens = list(seq.prompt_token_ids) + [self.eos] * (padded_len - real_len)
        tokens = hax.NamedArray(jnp.array(padded_tokens, dtype=jnp.int32), axes=(pos_axis,))
        seq_named = hax.named([seq.seq_id], "seq")
        temps = hax.full((), seq.sampling_params.temperature, dtype=jnp.float32)
        key = jrandom.PRNGKey(0)
        tok, page_table, cache = do_prefill(self.model, cache, page_table, tokens, self.sampler, seq_named, temps, key)
        return int(tok.array), cache, page_table

    def _decode(self, seq: Sequence, cache, page_table, step: int):
        prev_token = jnp.array([seq.last_token], dtype=jnp.int32)
        seq_named = hax.named([seq.seq_id], "seq")
        temps = hax.full((), seq.sampling_params.temperature, dtype=jnp.float32)
        key = jrandom.PRNGKey(step)
        start = jnp.array(step, dtype=jnp.int32)
        tok, page_table, cache = do_generate(
            self.model, cache, page_table, prev_token, self.sampler, seq_named, start, temps, key
        )
        return int(tok.array), cache, page_table

    def generate(
        self, prompts: List[str] | List[List[int]], sampling_params: SamplingParams | List[SamplingParams]
    ) -> List[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        prompt_ids_list: List[List[int]] = [
            self.tokenizer.encode(p, add_special_tokens=False) if isinstance(p, str) else list(p)
            for p in prompts
        ]

        if len(prompt_ids_list) > MAX_SEQS:
            raise ValueError(f"Too many prompts: got {len(prompt_ids_list)}, max {MAX_SEQS}")

        max_prompt = max(len(p) for p in prompt_ids_list)
        max_tokens = max(sp.max_tokens for sp in sampling_params)
        page_size = _round_preferred(max_prompt + max_tokens)
        page_table = PageTable.init(
            max_pages=MAX_SEQS,
            max_seqs=MAX_SEQS,
            page_size=page_size,
            max_pages_per_seq=1,
        )
        seq_ids = []
        for _ in prompt_ids_list:
            page_table, seq_id = page_table.assign_seq_id_to_seq()
            seq_ids.append(seq_id)
        cache = self.model.initial_cache(page_table, dtype=self.trainer_cfg.mp.compute_dtype)

        scheduler = Scheduler(self.eos)
        seq_objs = []
        for p, sp, seq_id in zip(prompt_ids_list, sampling_params, seq_ids):
            seq = Sequence(p, sp, seq_id=seq_id)
            seq_objs.append(seq)
            scheduler.add(seq)

        outputs = {}
        while not scheduler.is_finished():
            seqs, is_prefill = scheduler.schedule()
            token_ids = []
            for seq in seqs:
                if is_prefill and seq.status is SequenceStatus.WAITING:
                    tok, cache, page_table = self._prefill(seq, cache, page_table)
                    seq.status = SequenceStatus.RUNNING
                else:
                    tok, cache, page_table = self._decode(seq, cache, page_table, len(seq))
                token_ids.append(tok)
            scheduler.postprocess(seqs, token_ids)
            for seq in seqs:
                if seq.is_finished:
                    outputs[seq.seq_id] = seq.token_ids
        return [
            {"text": self.tokenizer.decode(out, skip_special_tokens=True), "token_ids": out}
            for _, out in sorted(outputs.items())
        ]


class LLM(LLMEngine):
    pass
