# Draft Design Document for Inference

> **Note**: This document contains both the technical design and project implementation tasks. For the current project status and milestones, see [`.agents/projects/inference.md`](/.agents/projects/inference.md).

## References

* [JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/blob/b282713880943cebe7183815918fb7dd60922b14/llama3/llama3_jax/model.py) <-- pretty basic kv cache, but easy to follow
* [Fancier JAX LLM Example](https://github.com/jax-ml/jax-llm-examples/pull/22/files)

### MaxText

* [Page Manager](https://github.com/AI-Hypercomputer/maxtext/blob/eac885edb371e6141a2bb784f9060f816ce17b23/MaxText/inference/page_manager.py)
* [Paged Attention](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/inference/paged_attention.py#L298)
* [Jetstream](https://github.com/AI-Hypercomputer/JetStream) <-- session management and such
* [MaxEngine](https://github.com/AI-Hypercomputer/maxtext/blob/main/MaxText/maxengine.py) <-- maxtext impl of jetstream protocol

### EasyDel

* [VSurge](https://github.com/erfanzar/EasyDeL/tree/main/easydel/inference/vsurge)

### JAX Repo

- [Ragged Paged Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/ragged_paged_attention/kernel.py)

### Nano-VLLM

torch, not much interesting here given jetstream. focused on batch inference I think?

* https://github.com/GeeeekExplorer/nano-vllm/

### SGLang

* [SGLang](https://github.com/sgl-project/sglang)

## Project Overview

**Goal**: Expose Levanter models via an OpenAI-compatible HTTP API with streaming, batching, and scheduler-backed decoding. Reuse `src/levanter/main/sample_lm.py` and dependencies: `JitScheduler`, `DecodeState`, `PageTable`, `KvPageCache`, `Sampler`, tokenizer/HF loader, and `TrainerConfig` device mesh setup.

**Architecture notes**:
- Reuse `run_generation_loop`, `_one_round`, and `GenState` patterns from `src/levanter/main/sample_lm.py` but extract into a reusable service.
- Maintain decode state in a long-lived service object that owns: `JitScheduler`, `PageTable`, `KvPageCache`, `DecodeState`, `Sampler`, `tokenizer`, and the `LlamaLMHeadModel` (or other `LmHeadModel`).
- Requests enqueue prompt tokens into the service; a background loop performs prefill+decode rounds and streams tokens back to callers.
- Keep JIT-safety: no Python control flow in jitted sections; use existing named-jit and Equinox modules.

## Implementation Tasks

### 1. Extract reusable generation backend
- [ ] Create `src/levanter/inference/service.py` with `GenerationService`:
  - [ ] Initialization: device mesh (`TrainerConfig`), tokenizer loading, `_load_model`, `PageTable.init`, `model.initial_cache`, `JitScheduler.init`, `DecodeState.init`, `Sampler`.
  - [ ] Public APIs:
    - [ ] `generate(prompt: str | list[str], options) -> GeneratedResult`
    - [ ] `stream_generate(...)-> AsyncIterator[StreamChunk]`
  - [ ] Internals mirror `sample_lm` flow: enqueue prompts, `run_generation_loop`, `extract_outputs`, stop-seq handling, temperature, seed.
  - [ ] Accept per-request options: `max_tokens`, `temperature`, `top_p` (stub ok), `stop` (list[str]), `n` (later), `logprobs` (later).
- [ ] Refactor `run_generation_loop` logic into a method usable from both sync and async paths; keep jitted parts unchanged.
- [ ] Unit tests targeting service with a tiny config (CPU) to decode 1–2 tokens.

### 2. HTTP server scaffold
- [x] Add optional dependency group `serve`: FastAPI + Uvicorn (or Starlette). Optional: Pydantic v2.
- [x] New module `src/levanter/serve/min_server.py` exporting `app`.
- [x] Endpoints:
  - [x] `GET /healthz` returns 200 once model is loaded and warm.
  - [ ] `GET /v1/models` returns loaded model id(s).
  - [x] `POST /v1/completions` for text completion (prompt-based).
  - [ ] `POST /v1/chat/completions` for chat.
- [ ] Authentication: accept `Authorization: Bearer <token>` if `LEVANTER_API_KEY` set; otherwise open.
- [ ] Streaming: if `stream=true`, return SSE with `data: {delta...}` chunks and `data: [DONE]` per OpenAI semantics.

### 3. Request/response schema mapping
- [x] Define Pydantic models for requests/responses (subset of OpenAI spec):
  - [x] Completions: `prompt`, `model`, `max_tokens`, `temperature`, `stop`, `seed`.
  - [ ] Chat: `messages`, `model`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, `user`.
- [x] Response objects include `id`, `object`, `created`, `model`, `choices`, `usage`.
- [ ] Translate chat `messages` -> prompt via a simple, configurable chat template (default: Llama 3 style). Support `system` and `user` roles; ignore tools/functions initially.
- [ ] Compute `usage` based on tokenized prefix and generated tokens.

### 4. Scheduling, batching, and backpressure
- [ ] Async request queue that batches requests and feeds `JitScheduler`.
- [ ] Per-request state in `DecodeState` rows; enforce limits: `max_seq_len`, `pages_per_seq`, `page_size`, `max_new_tokens`.
- [ ] Backpressure: bound queue size and timeout; return 429/503 on saturation.
- [ ] Cancellation: detach sequence and `purge_queue_of_seq` on client disconnect.

### 5. Model loading, configuration, and CLI
- [x] Support either `checkpoint_path` or `hf_checkpoint` (mutually exclusive), same as `SampleLmConfig`.
- [x] Reuse `TrainerConfig` for device mesh and `mp` dtype; allow YAML config file.
- [x] CLI `python -m levanter.serve.min_server`:
  - [x] Args: `--hf-checkpoint`, `--checkpoint-path`, `--tokenizer`, `--port`, `--host`, `--seed`, `--log-level`, `--access-log`.
  - [ ] Warmup: perform one tiny decode to compile.

### 6. Observability
- [x] Structured logging for lifecycle and request events (start, end, errors, latency, tps).
- [ ] Optional Prometheus `/metrics` exporter with request counts, latencies, tokens/sec, compile time.
- [x] Error surfaces include truncated stack traces and 4xx/5xx mapping.

### 7. Testing
- [ ] Unit: schema validation, prompt templating, stop sequence application.
- [ ] Integration: spin up server with tiny CPU config; `POST /v1/completions` generating <= 4 tokens.
- [ ] Streaming: verify incremental deltas and `[DONE]` terminator.
- [ ] Load smoke: concurrent requests (N=4–8) exercise batching path.

### 8. Packaging & deployment
- [ ] Add optional `docs/design/Dockerfile` for serving (uv + runtime). Build stage compiles deps; runtime is slim.
- [ ] Example Kubernetes manifest (`infra/cluster/`) with resource requests, liveness/readiness probes.
- [ ] Document TPU/GPU flags and any XLA env needed.

### 9. Documentation & examples
- [ ] `README` snippet with curl examples for both endpoints.
- [ ] OpenAI SDK example:
  - Python: set `OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")`.
  - Node: `new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "sk-local" })`.
- [ ] Note unsupported options initially (function calling, tools, `response_format`, images, vision).

### 10. Stretch/Follow-ups
- [ ] `n>1` parallel sampling and multi-choice responses.
- [ ] `logprobs` surface and token-level details.
- [ ] Presence/frequency penalties; repetition penalty.
- [ ] `/v1/embeddings` if/when model supports.
- [ ] Tracing (OpenTelemetry) and richer metrics.

## Core Infrastructure Tasks

### Attention & Caching
- [x] Attention: do prefill for a batch of inputs
- [x] Attention: do decode for a batch of inputs
- [x] Attention: unit test that decode gives same output as prefill
- [x] simple sampler
- [x] initialize cache for a model (add a method to the lmheadmodel class?)
- [x] make a main to test the above
- [x] use paged attention for simple kvcache decoding (should be doable)
- [x] add support for ragged paged attention
- [x] stop sequences

### Offline Milestone
- [ ] make an offline batch inference that does continuous decoding
- [x] Update JIT scheduler to recognize sequences properly using `hax.where` with size/fill
- [x] Remove tokens from the queue based on sequence ID using a JIT-safe mask
- [x] Automatically slide tokens forward when removing entries
- [ ] Integrate stop sequence into decode state
- [ ] Implement max tokens per sequence limit
- [ ] Add option to record logprobs
- [ ] add swapping in new sequences into jit scheduler
- [ ] per-seq PRNG states inside the JIT scheduler
- [ ] Rename jit scheduler to something more appropriate (DecodeState?)
- [ ] Finish offline batch inference support

### LM Eval Harness milestone
- [ ] Support LM Eval Harness generation tasks
- [ ] Stop sequences
- [ ] max num tokens
- [ ] temperature

### Prefix Cache Integration
- [ ] Use radix tree draft to actually drive scheduling
- [ ] Enable automatic prefix caching
- [ ] When returning from JIT scheduler, insert allocated pages into the radix cache. careful to handle the case where there were the sampled tokens already happen to exist.
- [ ] Radix cache must return freed pages when a sequence is released
- [ ] Manage list of free pages outside the radix cache
- [ ] Double check radix LRU evict

### Optimization Ideas
- [ ] microoptimize jit dispatch
- [ ] Investigate why tensor parallelism doesn't work for logits / vocab mat mul
- [ ] Avoid computing full logit matrix during prefill unless `logprobs` are requested
- [ ] figure out why the profiler isn't giving me anything useful

### Sample LM Integration
- [ ] expose a free list of pages in `PageTable`
- [ ] allocate pages from the free list inside the generation loop
- [x] store partial sequences in `DecodeState` instead of `JitScheduler`
- [x] check `DecodeState.is_finished` during generation
- [ ] remove `PageTable` from the core decoding loop
- [ ] integrate any remaining pieces needed for `sample_lm`

## File/Code references
- `src/levanter/main/sample_lm.py`: `SampleLmConfig`, `_load_model`, `GenState`, `run_generation_loop`, `_one_round`, `extract_outputs`.
- `src/levanter/inference/jit_scheduler.py`: `JitScheduler`, `DecodeState`, `SeqDecodingParams`.
- `levanter.inference.page_table.PageTable`, `levanter.layers.attention.KvPageCache`.
- `levanter.layers.sampler.Sampler`.
- `levanter.models.llama.LlamaLMHeadModel` and `levanter.models.lm_model.LmHeadModel`.
- `levanter.compat.hf_checkpoints.HFCheckpointConverter`, `load_tokenizer`.
- `levanter.trainer.TrainerConfig` and `levanter.utils.jax_utils.use_cpu_device`.

## Acceptance criteria
- Can run: `python -m levanter.serve.min_server --hf-checkpoint <org/model> --tokenizer <tok> --port 8000` and:
  - [x] `GET /healthz` returns ready status.
  - [x] `POST /v1/completions` returns a valid OpenAI response.
  - [ ] `GET /v1/models` returns model id.
  - [ ] `POST /v1/chat/completions` returns a valid OpenAI response; streaming works with SSE; `usage` is populated.
  - [ ] Concurrency: 4 parallel requests; batching occurs; no crashes; graceful shutdown.
