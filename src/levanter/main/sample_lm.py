# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import asyncio
import collections
import collections.abc
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Union

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import uvicorn
from fastapi import FastAPI, HTTPException
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from pydantic import BaseModel

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.service import GenerationService, GenerationServiceConfig, Request
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


# Pydantic models for OpenAI API service
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = 16
    temperature: float = 1.0
    stop: Optional[Union[List[str], str]] = None
    n: int = 1
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    stop: Optional[Union[List[str], str]] = None
    n: int = 1
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


@dataclass
class SampleLmConfig:
    """Configuration for inference service."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    # Inference service/memory layout configuration
    service: GenerationServiceConfig = field(default_factory=GenerationServiceConfig)

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Default generation parameters
    max_new_tokens: int = 16
    temperature: float = 0.7
    seed: int = 42


class InferenceRequest:
    """Internal request structure for the inference thread"""

    def __init__(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        stop_tokens: Optional[List[int]],
        seed: int,
        future: asyncio.Future,
        n_generations: int = 1,
        original_prompt: str = "",
    ):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_tokens = stop_tokens
        self.seed = seed
        self.future = future
        self.n_generations = n_generations
        self.original_prompt = original_prompt


# A callback which replaces the current model.
WeightSource = collections.abc.Callable[[LmHeadModel], LmHeadModel]


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
        converter = converter.replaced(
            reference_checkpoint=config.hf_checkpoint, tokenizer=load_tokenizer(config.tokenizer)
        )
        model = converter.load_pretrained(
            config.model.model_type, ref=config.hf_checkpoint, dtype=config.trainer.mp.compute_dtype
        )
        return model


class InferenceContext:
    """Background thread that manages the GenerationService and processes requests"""

    def __init__(self, model: LmHeadModel, tokenizer, service: GenerationService, config):
        self.model = model
        self.tokenizer = tokenizer
        self.service = service
        self.config = config
        self.request_queue: queue.Queue[InferenceRequest] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.model_lock = threading.Lock()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._next_request_id = 0

    def start(self):
        """Start the inference thread"""
        self.thread.start()

    def shutdown(self):
        """Signal shutdown and wait for thread to finish"""
        self.shutdown_event.set()
        self.thread.join(timeout=10)

    def reload(self, weight_callback: WeightSource):
        """Reload the inference model using the given weight callback.

        If new weights are found, new requests are paused, existing requests are
        allowed to complete, and the new weights are loaded.
        """
        logger.info("New weights available, waiting for model lock...")
        lock_start_time = time.time()
        with self.model_lock:
            lock_wait_time = time.time() - lock_start_time
            logger.info(f"Acquired model lock after {lock_wait_time}, reloading weights...")

            start = time.time()
            self.model = weight_callback(self.model)
            elapsed = time.time() - start
            logger.info(f"Model reloaded in {elapsed:.2f}s")

    def submit_request(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        temperature: float,
        stop_tokens: Optional[List[int]],
        seed: int,
        future: asyncio.Future,
        n_generations: int = 1,
        original_prompt: str = "",
    ) -> str:
        """Submit a request to the inference queue"""
        request_id = f"req_{self._next_request_id}"
        self._next_request_id += 1

        request = InferenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
            seed=seed,
            future=future,
            n_generations=n_generations,
            original_prompt=original_prompt,
        )

        self.request_queue.put(request)
        return request_id

    def _inference_loop(self):
        """Main inference loop running in background thread"""
        logger.info("Inference thread started")

        while not self.shutdown_event.is_set():
            try:
                # Enqueue a batch of requests. Wait up to 0.1s to batch requests together.
                requests = []
                try:
                    req = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                requests.append(req)
                deadline = time.time() + 0.1
                while time.time() < deadline and len(requests) < self.config.service.max_seqs:
                    try:
                        requests.append(self.request_queue.get(timeout=deadline - time.time()))
                    except queue.Empty:
                        break

                with self.model_lock:
                    self._process_batch(requests)
            except Exception as e:
                logger.error(f"Error in inference loop: {e}", exc_info=True)

        logger.info("Inference thread shutting down")

    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of inference requests"""
        try:
            # Convert to GenerationService requests
            service_requests = []

            for i, req in enumerate(requests):
                # Create stop tokens if specified
                stop_ids = None
                if req.stop_tokens:
                    stop_ids = hax.named(
                        jnp.asarray(req.stop_tokens, dtype=jnp.int32), axis="position"
                    ).broadcast_axis({"stop_seq": 1})

                seq_params = SeqDecodingParams(
                    max_num_tokens=jnp.array(len(req.prompt_tokens) + req.max_tokens, dtype=jnp.int32),
                    stop_tokens=stop_ids,
                    temperature=jnp.array(req.temperature, dtype=jnp.float32),
                    key=jrandom.PRNGKey(req.seed if req.seed is not None else i),
                )

                service_req = Request(
                    prompt_tokens=req.prompt_tokens,
                    request_id=i,  # Use batch index as service request id
                    decode_params=seq_params,
                    n_generations=req.n_generations,
                )
                service_requests.append(service_req)

            # Generate responses
            logger.info(f"Processing batch of {len(requests)} requests")
            start_time = time.time()
            outputs, total_generated = self.service.generate(service_requests)
            duration = time.time() - start_time
            logger.info(f"Batch completed in {duration:.2f}s, generated {total_generated} tokens")

            # Return results to futures
            output_idx = 0
            for req in requests:
                try:
                    req_outputs = []
                    for _ in range(req.n_generations):
                        if output_idx < len(outputs):
                            # Decode tokens to text, excluding prompt
                            prompt_len = len(req.prompt_tokens)
                            generated_tokens = outputs[output_idx][prompt_len:]
                            text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            req_outputs.append(
                                {
                                    "text": text,
                                    "tokens": generated_tokens,
                                    "prompt_tokens": prompt_len,
                                    "completion_tokens": len(generated_tokens),
                                }
                            )
                            output_idx += 1
                        else:
                            logger.error(f"Missing output for request {req.request_id}")
                            req_outputs.append(
                                {
                                    "text": "",
                                    "tokens": [],
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                }
                            )

                    # Set the future result
                    req.future.get_loop().call_soon_threadsafe(req.future.set_result, req_outputs)
                except Exception as e:
                    logger.error(f"Error processing result for {req.request_id}: {e}")
                    req.future.get_loop().call_soon_threadsafe(req.future.set_exception, e)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            # Set exceptions on all futures
            for req in requests:
                try:
                    req.future.get_loop().call_soon_threadsafe(req.future.set_exception, e)
                except Exception:
                    pass


# Global inference thread instance
inference_context: Optional[InferenceContext] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global inference_context

    # Startup
    config = getattr(app.state, "config", None)
    if config is None:
        raise RuntimeError("Configuration not set. Call initialize_service() first.")

    logger.info("Initializing inference service...")

    tok_string: str | None = config.tokenizer
    if config.tokenizer is None:
        if config.hf_checkpoint is not None:
            tok_string = config.hf_checkpoint.model_name_or_path

    if tok_string is None:
        raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")

    tokenizer = load_tokenizer(tok_string)
    key = jrandom.PRNGKey(config.seed)

    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        model = _load_model(config, Vocab, key=key)
        assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

        service = GenerationService.from_model_with_config(model=model, tokenizer=tokenizer, config=config.service)

        # Create and start inference thread
        inference_context = InferenceContext(model, tokenizer, service, config)
        inference_context.start()

        logger.info("Inference service initialized and ready")

    yield

    # Shutdown
    if inference_context:
        logger.info("Shutting down inference thread...")
        inference_context.shutdown()
        logger.info("Inference thread shut down")


# FastAPI app
app = FastAPI(title="Levanter Inference Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "levanter-inference"}


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    """Create a text completion using OpenAI API format"""
    global inference_context
    if not inference_context:
        raise HTTPException(status_code=503, detail="Inference service not initialized")

    try:
        # Handle both string and list prompts
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        # Process stop sequences
        stop_tokens = None
        if request.stop:
            if isinstance(request.stop, str):
                stop_list = [request.stop]
            else:
                stop_list = request.stop

            # Tokenize stop sequences
            stop_tokens = []
            for stop in stop_list:
                stop_ids = inference_context.tokenizer(stop, add_special_tokens=False)["input_ids"]
                if stop_ids:
                    stop_tokens.extend(stop_ids)

        # Create futures for all prompts
        futures = []
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, prompt in enumerate(prompts):
            # Tokenize prompt
            prompt_tokens = inference_context.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            total_prompt_tokens += len(prompt_tokens)

            # Create future for this request
            future: asyncio.Future = asyncio.Future()
            futures.append(future)

            # Submit to inference thread
            inference_context.submit_request(
                prompt_tokens=prompt_tokens,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_tokens=stop_tokens,
                seed=request.seed if request.seed is not None else 42,
                future=future,
                n_generations=request.n,
                original_prompt=prompt,
            )

        # Wait for all results
        results = await asyncio.gather(*futures)

        # Format responses
        choice_idx = 0
        for prompt_idx, result in enumerate(results):
            for generation in result:
                choices.append(CompletionChoice(text=generation["text"], index=choice_idx, finish_reason="stop"))
                total_completion_tokens += generation["completion_tokens"]
                choice_idx += 1

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Create a chat completion using OpenAI API format"""
    global inference_context
    if not inference_context:
        raise HTTPException(status_code=503, detail="Inference service not initialized")

    try:
        # Convert messages to prompt using tokenizer's chat template
        messages = [msg.dict() for msg in request.messages]

        # Apply chat template
        try:
            prompt_tokens = inference_context.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception as e:
            # Fallback: simple concatenation if template fails
            logger.warning(f"Chat template failed, using fallback: {e}")
            prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt_tokens = inference_context.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Process stop sequences
        stop_tokens = None
        if request.stop:
            if isinstance(request.stop, str):
                stop_list = [request.stop]
            else:
                stop_list = request.stop

            stop_tokens = []
            for stop in stop_list:
                stop_ids = inference_context.tokenizer(stop, add_special_tokens=False)["input_ids"]
                if stop_ids:
                    stop_tokens.extend(stop_ids)

        # Create future and submit request
        future: asyncio.Future = asyncio.Future()
        inference_context.submit_request(
            prompt_tokens=prompt_tokens,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_tokens=stop_tokens,
            seed=request.seed if request.seed is not None else 42,
            future=future,
            n_generations=request.n,
        )

        # Wait for result
        results = await future

        # Format response
        choices = []
        total_completion_tokens = 0

        for i, generation in enumerate(results):
            choices.append(
                ChatChoice(
                    index=i,
                    message=ChatMessage(role="assistant", content=generation["text"]),
                    finish_reason="stop",
                )
            )
            total_completion_tokens += generation["completion_tokens"]

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=total_completion_tokens,
                total_tokens=len(prompt_tokens) + total_completion_tokens,
            ),
        )

    except Exception as e:
        logger.error(f"Error in chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def initialize_service(config: SampleLmConfig):
    """Initialize the service with configuration"""
    levanter.initialize(config)
    app.state.config = config


def main(config: SampleLmConfig):
    """Start the FastAPI inference server"""
    initialize_service(config)

    logger.info(f"Starting Levanter inference server on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    levanter.config.main(main)()
