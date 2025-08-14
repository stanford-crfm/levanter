from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import haliax
import haliax as hax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.jit_scheduler import JitScheduler, DecodeState, SeqDecodingParams
from levanter.inference.page_table import PageTable
from levanter.inference.utils import INVALID
from levanter.layers.attention import KvPageCache
from levanter.layers.sampler import Sampler
from levanter.models.lm_model import LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.main.sample_lm import run_generation_loop


@dataclass
class GenerationOptions:
    max_tokens: int = 16
    temperature: float = 0.7
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


class GenerationService:
    """
    Minimal generation service facade. For the initial milestone this is a stub that echoes the prompt,
    and will be wired to Levanter's JIT scheduler and decode loop in a follow-up edit.
    """

    def __init__(
        self,
        *,
        hf_checkpoint: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tokenizer: Optional[str] = None,
        seed: int = 0,
        trainer: Optional[TrainerConfig] = None,
    ):
        self._logger = logging.getLogger(__name__)
        self._hf_checkpoint = hf_checkpoint
        self._checkpoint_path = checkpoint_path
        self._tokenizer = tokenizer
        self._seed = seed
        self._trainer = trainer or TrainerConfig()

        # Runtime components
        self._tokenizer_obj = None
        self._model: Optional[LmHeadModel] = None
        self._sampler: Optional[Sampler] = None
        self._table: Optional[PageTable] = None
        self._cache: Optional[KvPageCache] = None
        self._sched: Optional[JitScheduler] = None
        self._decode_state: Optional[DecodeState] = None
        self._vocab_axis: Optional[Axis] = None

        self._last_error: Optional[str] = None

        try:
            self._logger.info("GenerationService initializing (hf_checkpoint=%s, checkpoint=%s)", self._hf_checkpoint, self._checkpoint_path)
            self._initialize()
            self._ready = True
            self._last_error = None
            self._logger.info("GenerationService initialized successfully: model=%s", type(self._model).__name__ if self._model else None)
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            self._logger.exception("Failed to initialize GenerationService: %s", e)
            self._ready = False

    def ready(self) -> bool:
        return self._ready

    @property
    def model_id(self) -> str:
        if self._hf_checkpoint is not None:
            return self._hf_checkpoint
        if self._checkpoint_path is not None:
            return self._checkpoint_path
        return "levanter-echo"

    def generate_once(self, prompt: str, options: GenerationOptions) -> str:
        assert self._model is not None and self._sampler is not None
        assert self._table is not None and self._cache is not None
        assert self._sched is not None and self._decode_state is not None
        assert self._tokenizer_obj is not None

        # Build stop ids if provided
        stop_ids_named = None
        if options.stop:
            # Only use the first stop sequence for now
            stop_str = options.stop[0]
            stop_ids = self._tokenizer_obj(stop_str, add_special_tokens=False)["input_ids"]
            if len(stop_ids) == 0:
                raise ValueError("Stop sequence must be non-empty")
            stop_ids_named = hax.named(np.asarray(stop_ids, dtype=np.int32), axis="position")

        # Tokenize prompt
        prompt_ids: list[int] = self._tokenizer_obj(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) == 0:
            return ""

        # Clone state for this request
        table = self._table
        cache = self._cache
        sched = self._sched
        decode_state = self._decode_state

        # assign a sequence slot
        table, seq_id = table.assign_seq_id_to_seq()

        # Per-sequence params
        max_total = int(len(prompt_ids) + max(0, options.max_tokens))
        temp = float(max(0.0, options.temperature))
        seed = int(options.seed if options.seed is not None else int(time.time()))
        key = jrandom.PRNGKey(seed)

        # expand stop tokens to the expected shape if provided
        if stop_ids_named is not None:
            stop_ids_broadcast = stop_ids_named.broadcast_axis({"stop_seq": 1})
        else:
            stop_ids_broadcast = None

        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(max_total, dtype=jnp.int32),
            stop_tokens=stop_ids_broadcast,
            temperature=jnp.array(temp, dtype=jnp.float32),
            key=jax.random.fold_in(key, seq_id),
        )

        # assign sequence tokens
        decode_state = decode_state.assign_seq(
            seq_id,
            seq_id,
            hax.full({"page": table.pages_per_seq}, INVALID, dtype=jnp.int32),
            hax.named(np.asarray(prompt_ids, dtype=jnp.int32), axis="position"),
            len(prompt_ids),
            seq_params,
        )

        # enqueue prompt tokens
        prompts_tokens_to_enqueue = np.asarray(prompt_ids, dtype=jnp.int32)
        if len(prompt_ids) > sched.max_queued_tokens:
            raise ValueError(
                f"Prompt too long ({len(prompt_ids)} tokens) for queue size {sched.max_queued_tokens}"
            )

        while len(prompts_tokens_to_enqueue) > sched.empty_queue_space:
            # free space with a short generation loop
            gen_state = _GenState(sched=sched, cache=cache, page_table=table, decode_state=decode_state)
            gen_state = run_generation_loop(gen_state, self._model, self._sampler, 64, 32)
            sched, cache, table, decode_state = gen_state.sched, gen_state.cache, gen_state.page_table, gen_state.decode_state
            _ = _extract_outputs(decode_state, [[]], [False])

        this_tokens = hax.named(prompts_tokens_to_enqueue, axis="position")
        seq_ids = hax.full_like(this_tokens, seq_id, dtype=jnp.int32)
        sched = sched.enqueue_tokens(this_tokens, seq_ids, prompts_tokens_to_enqueue.size)

        # One macro-prefill round
        gen_state = _GenState(sched=sched, cache=cache, page_table=table, decode_state=decode_state)
        gen_state = run_generation_loop(gen_state, self._model, self._sampler, 16, 1)
        sched, cache, table, decode_state = gen_state.sched, gen_state.cache, gen_state.page_table, gen_state.decode_state

        # Decode loop until finished
        finished = [False]
        outputs = [list(prompt_ids)]
        _extract_outputs(decode_state, outputs, finished)

        while not all(finished):
            gen_state = _GenState(sched=sched, cache=cache, page_table=table, decode_state=decode_state)
            gen_state = run_generation_loop(gen_state, self._model, self._sampler, 16, 8)
            sched, cache, table, decode_state = gen_state.sched, gen_state.cache, gen_state.page_table, gen_state.decode_state
            _extract_outputs(decode_state, outputs, finished)

        # Cleanup: free pages and reset local structures for next request
        table = table.free_pages(seq_id)
        decode_state = DecodeState.init(
            table.max_seqs,
            table.pages_per_seq,
            table.page_size,
            table.max_len_per_seq,
            max_stop_seqs=1,
        )
        sched = sched.cleared()

        # Persist updated components for subsequent requests
        self._table, self._cache, self._sched, self._decode_state = table, cache, sched, decode_state

        # Decode tokens to text
        seq_outputs = [tok for tok in outputs[0] if tok != self._tokenizer_obj.pad_token_id and tok != INVALID]
        text = self._tokenizer_obj.decode(seq_outputs, skip_special_tokens=True)
        return text

    # ---- Internal helpers ----
    def _initialize(self):
        if self._hf_checkpoint is None and self._checkpoint_path is None:
            raise ValueError("Must specify either hf_checkpoint or checkpoint_path")

        # Tokenizer
        tok_ref = self._tokenizer or self._hf_checkpoint
        if tok_ref is None:
            raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")
        self._tokenizer_obj = load_tokenizer(tok_ref)

        key = jrandom.key(self._seed)

        with self._trainer.device_mesh, hax.axis_mapping(self._trainer.compute_axis_mapping):
            vocab_size = len(self._tokenizer_obj)
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), self._trainer.compute_axis_mapping)
            self._vocab_axis = Vocab
            model = self._load_model(Vocab, key)
            assert isinstance(model, LmHeadModel), "Loaded model is not an LmHeadModel"
            self._model = model
            self._sampler = Sampler(Vocab)

            # Buffers
            table = PageTable.init(64, 1, 8, 32)
            cache = haliax.named_jit(model.initial_cache)(table, dtype=self._trainer.mp.compute_dtype)
            sched = JitScheduler.init(128)
            decode_state = DecodeState.init(
                table.max_seqs,
                table.pages_per_seq,
                table.page_size,
                table.max_len_per_seq,
                max_stop_seqs=1,
            )

            self._table, self._cache, self._sched, self._decode_state = table, cache, sched, decode_state

    def _load_model(self, Vocab: Axis, key) -> LmHeadModel:
        if self._checkpoint_path is None and self._hf_checkpoint is None:
            raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
        if self._checkpoint_path is not None and self._hf_checkpoint is not None:
            raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

        mp = self._trainer.mp
        if self._checkpoint_path is not None:
            # TODO: Support local Levanter checkpoints with a provided model config
            raise NotImplementedError("Local checkpoint loading not yet supported in GenerationService")
        else:
            if self._hf_checkpoint is None:
                raise ValueError("hf_checkpoint must be specified")
            converter: HFCheckpointConverter = HFCheckpointConverter.from_hf(RepoRef.from_string(self._hf_checkpoint))
            converter = converter.replaced(reference_checkpoint=self._hf_checkpoint, tokenizer=self._tokenizer_obj)
            model = converter.load_pretrained(
                converter.config_from_hf_checkpoint().model_type,
                ref=self._hf_checkpoint,
                axis_mapping=self._trainer.compute_axis_mapping,
                dtype=mp.compute_dtype,
            )
            return model  # type: ignore


@dataclasses.dataclass
class _GenState(eqx.Module):
    sched: JitScheduler
    cache: KvPageCache
    page_table: PageTable
    decode_state: DecodeState


def _extract_outputs(decode_state: DecodeState, outputs, finished):
    # Minimal extraction for our single sequence case (adapted from sample_lm.extract_outputs)
    num_seqs = decode_state.max_seqs
    tokens = jax.device_get(decode_state.tokens.array)
    num_tokens = jax.device_get(decode_state.num_tokens.array)
    this_finished = jax.device_get(decode_state.is_finished(jnp.arange(num_seqs)))
    for seq_id in range(num_seqs):
        current_num_tokens = len(outputs[seq_id])
        new_num_tokens = num_tokens[seq_id]
        count_to_extract = new_num_tokens - current_num_tokens
        if finished[seq_id]:
            continue
        if this_finished[seq_id]:
            finished[seq_id] = True
        if count_to_extract <= 0:
            continue
        seq_tokens = tokens[seq_id, current_num_tokens:new_num_tokens]
        outputs[seq_id].extend(int(x) for x in seq_tokens)
    return
