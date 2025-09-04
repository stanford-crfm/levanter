from __future__ import annotations

import dataclasses
import threading
import logging
import time
import asyncio
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Deque
from collections import deque

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


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    options: GenerationOptions
    response_callback: Callable[[GenerationResult], None]
    request_id: str


@dataclass
class GenerationResult:
    """Result of a text generation request."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str = "stop"


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

        # Queue-based architecture
        self._request_queue: queue.Queue[GenerationRequest] = queue.Queue()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._pending: Deque[GenerationRequest] = deque()

        # Shared continuous-batch state
        self._gen_state: Optional[_GenState] = None
        self._outputs: list[list[int]] = []
        self._finished: list[bool] = []
        self._active: Dict[int, Dict[str, Any]] = {}

        try:
            self._logger.info("GenerationService initializing (hf_checkpoint=%s, checkpoint=%s)", self._hf_checkpoint, self._checkpoint_path)
            self._initialize()
            self._start_scheduler()
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
    def tokenizer(self):
        """Expose the underlying HF tokenizer instance."""
        return self._tokenizer_obj

    def apply_chat_template(self, messages: list[dict[str, str]], *, add_generation_prompt: bool = True) -> str:
        """Render chat messages to a single prompt string using the HF tokenizer's chat template
        if available; otherwise fall back to a simple format.

        Args:
            messages: list of {"role": str, "content": str}
            add_generation_prompt: whether to include the assistant prefix
        Returns:
            Rendered prompt string suitable for completion generation.
        """
        tok = self._tokenizer_obj
        # Prefer HF apply_chat_template if present
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(  # type: ignore[attr-defined]
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                # fall back below if something goes wrong
                pass

        # Fallback: simple Llama-style chat prompt
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
            else:
                parts.append(f"<|{role}|>\n{content}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def _start_scheduler(self):
        """Start the scheduler thread that processes requests from the queue."""
        self._scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self._scheduler_thread.start()
        self._logger.info("Scheduler thread started")

    def _scheduler_worker(self):
        """Continuous batching worker: assign, prefill, decode, and finalize multiple requests."""
        assert self._model is not None and self._sampler is not None
        assert self._table is not None and self._cache is not None and self._sched is not None and self._decode_state is not None

        # Initialize shared gen state
        self._gen_state = _GenState(sched=self._sched, cache=self._cache, page_table=self._table, decode_state=self._decode_state)
        self._outputs = [[] for _ in range(self._gen_state.decode_state.max_seqs)]
        self._finished = [False for _ in range(self._gen_state.decode_state.max_seqs)]
        self._active = {}

        while not self._shutdown_event.is_set():
            try:
                # Drain new requests if available
                drained = 0
                while drained < 32:
                    try:
                        req = self._request_queue.get_nowait()
                        # Mark as taken from the queue
                        self._request_queue.task_done()
                        self._pending.append(req)
                        drained += 1
                    except queue.Empty:
                        break

                # Try to assign pending requests to free seq slots
                self._assign_pending_requests()

                # If nothing active and nothing queued, wait briefly
                if len(self._active) == 0 and self._gen_state.sched.num_queued_tokens == 0 and len(self._pending) == 0:
                    time.sleep(0.005)
                    continue

                # Run a few decode/prefill steps
                self._gen_state = run_generation_loop(
                    self._gen_state,
                    self._model,
                    self._sampler,
                    64,
                    4,
                )

                # Extract outputs and update finished flags
                _extract_outputs(self._gen_state.decode_state, self._outputs, self._finished)

                # Finalize any finished sequences
                self._finalize_finished_sequences()

            except Exception as e:
                self._logger.exception("Unexpected error in scheduler loop: %s", e)
                time.sleep(0.05)

    def shutdown(self):
        """Shutdown the service and scheduler thread."""
        self._shutdown_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
        self._logger.info("GenerationService shutdown complete")

    # ---- Continuous batching helpers ----
    def _assign_pending_requests(self):
        """Assign pending requests to available decode slots; enqueue their prompts."""
        assert self._gen_state is not None and self._tokenizer_obj is not None
        # Iterate over a snapshot of pending requests to avoid long holds
        i = 0
        while i < len(self._pending):
            req = self._pending[0]
            # Try to allocate a seq slot
            new_table, seq_id = self._gen_state.page_table.assign_seq_id_to_seq()
            if int(seq_id) == INVALID:
                # No free slots; need to run decode more to free
                break

            # We have a slot; pop from pending
            self._pending.popleft()
            self._gen_state = dataclasses.replace(self._gen_state, page_table=new_table)

            # Tokenize prompt
            enc = self._tokenizer_obj(req.prompt, add_special_tokens=False)
            prompt_ids: list[int] = enc["input_ids"]
            if not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)

            # Build stop tokens
            stop_ids_named = None
            if req.options.stop:
                stop_str = req.options.stop[0]
                stop_tok = self._tokenizer_obj(stop_str, add_special_tokens=False)["input_ids"]
                if len(stop_tok) == 0:
                    # if invalid stop, ignore
                    stop_ids_named = None
                else:
                    stop_ids_named = hax.named(np.asarray(stop_tok, dtype=np.int32), axis="position").broadcast_axis({"stop_seq": 1})

            # Seq params
            max_total_tokens = int(len(prompt_ids) + req.options.max_tokens)
            temperature = float(req.options.temperature)
            base_key = jrandom.PRNGKey(req.options.seed if req.options.seed is not None else int(time.time()))
            seq_key = jrandom.fold_in(base_key, int(seq_id))
            seq_params = SeqDecodingParams(
                max_num_tokens=jnp.array(max_total_tokens, dtype=jnp.int32),
                stop_tokens=stop_ids_named,
                temperature=jnp.array(temperature, dtype=jnp.float32),
                key=seq_key,
            )

            # Assign into decode state
            self._gen_state = dataclasses.replace(
                self._gen_state,
                decode_state=self._gen_state.decode_state.assign_seq(
                    int(seq_id),
                    int(seq_id),
                    hax.full({"page": self._gen_state.page_table.pages_per_seq}, INVALID, dtype=jnp.int32),
                    hax.named(np.asarray(prompt_ids, dtype=np.int32), axis="position"),
                    len(prompt_ids),
                    seq_params,
                ),
            )

            # Clear previous outputs for this slot and mark unfinished
            self._outputs[int(seq_id)] = list(prompt_ids)
            self._finished[int(seq_id)] = False

            # Ensure there's room in the queue, otherwise we'll rely on the next gen loop
            prompt_arr = np.asarray(prompt_ids, dtype=np.int32)
            if prompt_arr.size > self._gen_state.sched.max_queued_tokens:
                # Extremely long prompt; truncate or error. For now, error to keep simple.
                self._fail_request(req, f"Prompt too long ({prompt_arr.size}) for queue {self._gen_state.sched.max_queued_tokens}")
                # Free seq id
                self._gen_state = dataclasses.replace(self._gen_state, page_table=self._gen_state.page_table.free_pages(int(seq_id)))
                continue

            # Ensure there is enough space in the queue to enqueue the full prompt
            prompt_arr = np.asarray(prompt_ids, dtype=np.int32)
            while prompt_arr.size > self._gen_state.sched.empty_queue_space:
                # run a short loop to free up queue space
                self._gen_state = run_generation_loop(
                    self._gen_state,
                    self._model,
                    self._sampler,
                    64,
                    2,
                )
                _extract_outputs(self._gen_state.decode_state, self._outputs, self._finished)

            # Enqueue prompt tokens
            this_tokens = hax.named(prompt_arr, axis="position")
            seq_ids = hax.full_like(this_tokens, int(seq_id), dtype=jnp.int32)
            self._gen_state = dataclasses.replace(
                self._gen_state,
                sched=self._gen_state.sched.enqueue_tokens(this_tokens, seq_ids, prompt_arr.size),
            )

            # Track active request data
            self._active[int(seq_id)] = {
                "request": req,
                "prompt_len": len(prompt_ids),
                "max_total_tokens": max_total_tokens,
            }

        # Done assigning available ones

    def _finalize_finished_sequences(self):
        """Send callbacks for finished sequences and reclaim resources."""
        assert self._gen_state is not None and self._tokenizer_obj is not None
        done_seq_ids = []
        for seq_id, meta in list(self._active.items()):
            if self._finished[seq_id] or len(self._outputs[seq_id]) >= meta["max_total_tokens"]:
                done_seq_ids.append(seq_id)

        for seq_id in done_seq_ids:
            meta = self._active.pop(seq_id)
            req: GenerationRequest = meta["request"]
            prompt_len: int = meta["prompt_len"]
            toks = self._outputs[seq_id]

            # Strip padding/invalids
            pad_id = getattr(self._tokenizer_obj, "pad_token_id", None)
            clean = [t for t in toks if (pad_id is None or t != pad_id) and t != INVALID]
            completion_ids = clean[prompt_len:]
            # Return text that includes the original prompt prefix so tests can assert startswith(prompt)
            text = self._tokenizer_obj.decode(clean, skip_special_tokens=True)

            result = GenerationResult(
                text=text,
                prompt_tokens=prompt_len,
                completion_tokens=len(completion_ids),
                total_tokens=len(clean),
                finish_reason="stop" if self._finished[seq_id] else "length",
            )

            try:
                req.response_callback(result)
            finally:
                # Free resources
                self._gen_state = dataclasses.replace(self._gen_state, page_table=self._gen_state.page_table.free_pages(int(seq_id)))
                self._gen_state = dataclasses.replace(self._gen_state, sched=self._gen_state.sched.purge_queue_of_seq(int(seq_id)))
                # reset outputs and flag for reuse
                self._outputs[seq_id] = []
                self._finished[seq_id] = False

    def _fail_request(self, req: GenerationRequest, msg: str):
        self._logger.error("Request %s failed: %s", req.request_id, msg)
        try:
            req.response_callback(
                GenerationResult(text="", prompt_tokens=0, completion_tokens=0, total_tokens=0, finish_reason="error")
            )
        except Exception:
            pass

    @property
    def model_id(self) -> str:
        if self._hf_checkpoint is not None:
            return self._hf_checkpoint
        if self._checkpoint_path is not None:
            return self._checkpoint_path
        return "levanter-echo"

    def _generate_once_internal(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        """
        Internal method that performs the actual generation.
        This runs in the scheduler thread.
        """
        assert self._model is not None and self._sampler is not None and self._tokenizer_obj is not None

        # Tokenize prompt
        enc = self._tokenizer_obj(prompt, add_special_tokens=False)
        prompt_ids: list[int] = enc["input_ids"]
        if not isinstance(prompt_ids, list):
            # some tokenizers may return np arrays
            prompt_ids = list(prompt_ids)

        # Build stop sequence tokens (single stop seq supported for now)
        stop_ids_named = None
        if options.stop:
            # Support only the first stop sequence for now
            stop_str = options.stop[0]
            stop_tok = self._tokenizer_obj(stop_str, add_special_tokens=False)["input_ids"]
            if len(stop_tok) == 0:
                raise ValueError("Stop sequence must be non-empty after tokenization")
            stop_ids_named = hax.named(np.asarray(stop_tok, dtype=np.int32), axis="position")

        # Fresh per-request state (simple, avoids cross-request interference)
        # Dimensions based on the service-initialized capacities
        assert self._table is not None and self._cache is not None and self._sched is not None and self._decode_state is not None
        table = PageTable.init(
            self._table.num_pages,
            self._table.max_seqs,
            self._table.page_size,
            self._table.pages_per_seq,
        )
        cache = haliax.named_jit(self._model.initial_cache)(table, dtype=self._trainer.mp.compute_dtype)
        sched = self._sched.cleared()
        decode_state = DecodeState.init(
            table.max_seqs,
            table.pages_per_seq,
            table.page_size,
            table.max_len_per_seq,
            max_stop_seqs=1,
        )
        gen_state = _GenState(sched=sched, cache=cache, page_table=table, decode_state=decode_state)

        # Assign a sequence and set decoding parameters
        page_table, seq_id = gen_state.page_table.assign_seq_id_to_seq()
        gen_state = dataclasses.replace(gen_state, page_table=page_table)

        max_total_tokens = int(len(prompt_ids) + options.max_tokens)
        # Broadcast stop ids across stop_seq dimension if present
        if stop_ids_named is not None:
            stop_ids_named = stop_ids_named.broadcast_axis({"stop_seq": 1})

        # Temperature and PRNG
        temperature = float(options.temperature)
        base_key = jrandom.PRNGKey(options.seed if options.seed is not None else int(time.time()))
        seq_key = jrandom.fold_in(base_key, int(seq_id))

        seq_params = SeqDecodingParams(
            max_num_tokens=jnp.array(max_total_tokens, dtype=jnp.int32),
            stop_tokens=stop_ids_named,
            temperature=jnp.array(temperature, dtype=jnp.float32),
            key=seq_key,
        )

        # Initialize decode state for this sequence
        gen_state = dataclasses.replace(
            gen_state,
            decode_state=gen_state.decode_state.assign_seq(
                int(seq_id),
                int(seq_id),
                hax.full({"page": gen_state.page_table.pages_per_seq}, INVALID, dtype=jnp.int32),
                hax.named(np.asarray(prompt_ids, dtype=np.int32), axis="position"),
                len(prompt_ids),
                seq_params,
            ),
        )

        # Enqueue prompt tokens (may need to run the loop to free space)
        prompt_arr = np.asarray(prompt_ids, dtype=np.int32)
        if prompt_arr.size > gen_state.sched.max_queued_tokens:
            raise ValueError(
                f"Prompt too long ({prompt_arr.size} tokens) for scheduler queue of {gen_state.sched.max_queued_tokens}"
            )

        target_len = prompt_arr.size
        # Run gen loop to free space if necessary
        while target_len > gen_state.sched.empty_queue_space:
            gen_state = run_generation_loop(
                gen_state,
                self._model,
                self._sampler,
                64,
                32,
            )

        # Actually enqueue
        this_tokens = hax.named(prompt_arr, axis="position")
        seq_ids = hax.full_like(this_tokens, int(seq_id), dtype=jnp.int32)
        gen_state = dataclasses.replace(
            gen_state,
            sched=gen_state.sched.enqueue_tokens(this_tokens, seq_ids, prompt_arr.size),
        )

        # One macro prefill round to kick things off
        gen_state = run_generation_loop(gen_state, self._model, self._sampler, 16, 1)

        # Extract outputs and then autoregressive loop until finished or length cap
        finished = [False] * gen_state.decode_state.max_seqs
        outputs = [list(prompt_ids) for _ in range(gen_state.decode_state.max_seqs)]
        _extract_outputs(gen_state.decode_state, outputs, finished)

        # Continue until this sequence is finished
        while not finished[int(seq_id)]:
            gen_state = run_generation_loop(
                gen_state,
                self._model,
                self._sampler,
                1,  # small batch is fine for single seq
                8,
            )
            _extract_outputs(gen_state.decode_state, outputs, finished)

            # Safety: break if we've reached max tokens to avoid infinite loop
            if len(outputs[int(seq_id)]) >= max_total_tokens:
                break

        # Collect and decode the generated continuation (exclude the prompt)
        seq_outputs = outputs[int(seq_id)]
        # Strip padding/invalids
        seq_outputs = [tok for tok in seq_outputs if tok != self._tokenizer_obj.pad_token_id and tok != INVALID]
        completion_token_ids = seq_outputs[len(prompt_ids):]
        text = self._tokenizer_obj.decode(completion_token_ids, skip_special_tokens=True)

        prompt_tokens = len(prompt_ids)
        completion_tokens = len(completion_token_ids)
        total_tokens = prompt_tokens + completion_tokens

        finish_reason = "stop" if finished[int(seq_id)] else "length"

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    def generate_once(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        """
        Generate text from a prompt using the queue-based scheduler.

        This method is synchronous but internally uses a queue to avoid blocking.
        For true async operation, use generate_once_async instead.
        """
        import uuid

        # Create a future to wait for the result
        result_future = threading.Event()
        result_container = {"result": None}

        def callback(result: GenerationResult):
            result_container["result"] = result  # type: ignore
            result_future.set()

        # Create and queue the request
        request = GenerationRequest(
            prompt=prompt,
            options=options,
            response_callback=callback,
            request_id=str(uuid.uuid4())
        )

        self._request_queue.put(request)

        # Wait for the result
        result_future.wait()
        return result_container["result"]  # type: ignore

    async def generate_once_async(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        """
        Async version of generate_once that returns a future.
        """
        import uuid

        # Create a future to wait for the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def callback(result: GenerationResult):
            if not future.done():
                loop.call_soon_threadsafe(future.set_result, result)

        # Create and queue the request
        request = GenerationRequest(
            prompt=prompt,
            options=options,
            response_callback=callback,
            request_id=str(uuid.uuid4())
        )

        self._request_queue.put(request)

        # Wait for the result
        return await future

    # ---- Internal helpers ----
    def _initialize(self):
        if self._hf_checkpoint is None and self._checkpoint_path is None:
            raise ValueError("Must specify either hf_checkpoint or checkpoint_path")

        # Tokenizer
        tok_ref = self._tokenizer or self._hf_checkpoint
        if tok_ref is None:
            raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")
        self._tokenizer_obj = load_tokenizer(tok_ref)

        key = jrandom.PRNGKey(self._seed)

        with self._trainer.device_mesh, hax.axis_mapping(self._trainer.compute_axis_mapping):
            vocab_size = len(self._tokenizer_obj)
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), self._trainer.compute_axis_mapping)
            self._vocab_axis = Vocab
            model = self._load_model(Vocab, key)
            assert isinstance(model, LmHeadModel), "Loaded model is not an LmHeadModel"
            self._model = model
            self._sampler = Sampler(Vocab)

            # Buffers
            # Conservative defaults allowing a few concurrent sequences
            table = PageTable.init(64, 4, 8, 32)
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
