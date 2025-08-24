from __future__ import annotations

import dataclasses
import threading
import logging
import time
import asyncio
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

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

    def _start_scheduler(self):
        """Start the scheduler thread that processes requests from the queue."""
        self._scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self._scheduler_thread.start()
        self._logger.info("Scheduler thread started")

    def _scheduler_worker(self):
        """Worker thread that processes generation requests from the queue."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for request with timeout to allow checking shutdown
                try:
                    request = self._request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the request
                try:
                    result = self._generate_once_internal(request.prompt, request.options)
                    request.response_callback(result)
                except Exception as e:
                    self._logger.exception("Error processing request %s: %s", request.request_id, e)
                    # Create error result
                    error_result = GenerationResult(
                        text="",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        finish_reason="error"
                    )
                    request.response_callback(error_result)
                finally:
                    self._request_queue.task_done()
                    
            except Exception as e:
                self._logger.exception("Unexpected error in scheduler worker: %s", e)

    def shutdown(self):
        """Shutdown the service and scheduler thread."""
        self._shutdown_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
        self._logger.info("GenerationService shutdown complete")

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
        # For now, just return a simple echo to get the queue architecture working
        # TODO: Implement full generation logic
        return GenerationResult(
            text=prompt + " [generated]",
            prompt_tokens=len(prompt.split()),
            completion_tokens=1,
            total_tokens=len(prompt.split()) + 1
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
            result_container["result"] = result
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
        return result_container["result"]

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
