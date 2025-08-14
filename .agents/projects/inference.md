## OAI-compatible Inference Server (Levanter)

Goal: Expose Levanter models via an OpenAI-compatible HTTP API with streaming, batching, and scheduler-backed decoding. Reuse `src/levanter/main/sample_lm.py` and dependencies: `JitScheduler`, `DecodeState`, `PageTable`, `KvPageCache`, `Sampler`, tokenizer/HF loader, and `TrainerConfig` device mesh setup.

### Milestone: Minimal single-turn completions (FastAPI)
- **Scope**: Implement only `POST /v1/completions` (no chat, no streaming, batch size = 1). Single request -> single output text. Use tokenizer + model via a simplified wrapper of `sample_lm` logic (prefill + decode loop), with `max_tokens`, `temperature`, and optional `stop` (string or list of strings). Return OpenAI completion response shape.
- **Out of scope**: SSE streaming, batching >1, usage tokens accounting beyond basic counts, top_p/top_k penalties, function/tool calls.

- [ ] Server scaffold
  - [x] Add optional dependency group `serve` (fastapi + uvicorn).
  - [x] Create `src/levanter/serve/min_server.py` exposing `app` and a `main()` to run uvicorn.
  - [x] Endpoint: `POST /v1/completions` with minimal OpenAI-compatible schema.
  - [x] Health: `GET /healthz` returns 200 when model is ready.

- [ ] Minimal model wrapper
  - [ ] `GenerationService` (single-seq) that loads tokenizer + model via `_load_model` and sets up `PageTable`, `KvPageCache`, `JitScheduler`, `DecodeState`, `Sampler`.
  - [ ] `generate_once(prompt, max_tokens, temperature, stop) -> text` using the same enqueue + `run_generation_loop` + `extract_outputs` flow as `sample_lm`, but limited to a single sequence and a blocking loop until finish or `max_tokens`.
  - [ ] Convert `stop` strings to token sequences with tokenizer and pass via `SeqDecodingParams`.

- [ ] Request/response models (subset of OpenAI)
  - [x] Request: `{ model: str, prompt: str, max_tokens?: int, temperature?: float, stop?: str | string[], seed?: int }`
  - [x] Response: `{ id, object: "text_completion", created, model, choices: [{ index, text, finish_reason }], usage: { prompt_tokens, completion_tokens, total_tokens } }`

- [ ] CLI
  - [x] `python -m levanter.serve.min_server host=0.0.0.0 port=8000 hf_checkpoint=<org/model> tokenizer=<org/model>` (draccus)
  - [ ] Warmup one-token decode on startup to trigger JIT.

- [ ] Acceptance criteria
  - [ ] `curl -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"<id>","prompt":"Hello","max_tokens":8}'` returns a valid response with non-empty `choices[0].text`.
  - [ ] End-to-end runs on CPU with a tiny HF model; no crashes; initial JIT compile acknowledged in logs.

Notes: This milestone deliberately avoids concurrency and streaming to minimize moving parts while validating the integration of tokenizer, model, scheduler, and response schema.

### Deliverables
- [ ] HTTP server exposing OpenAI-compatible endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/healthz`
- [ ] Streaming SSE for chat/completions (`stream=true`)
- [ ] Batching and continuous decoding using `JitScheduler`
- [ ] Model loading from HF repo or local checkpoint; tokenizer wiring
- [ ] Minimal CLI to launch server; config file + env overrides
- [ ] Basic observability (logs, metrics) and graceful shutdown
- [ ] Tests: unit + simple e2e with a tiny model on CPU
- [ ] Quickstart docs and OpenAI SDK example

### Architecture notes
- Reuse `run_generation_loop`, `_one_round`, and `GenState` patterns from `src/levanter/main/sample_lm.py` but extract into a reusable service.
- Maintain decode state in a long-lived service object that owns: `JitScheduler`, `PageTable`, `KvPageCache`, `DecodeState`, `Sampler`, `tokenizer`, and the `LlamaLMHeadModel` (or other `LmHeadModel`).
- Requests enqueue prompt tokens into the service; a background loop performs prefill+decode rounds and streams tokens back to callers.
- Keep JIT-safety: no Python control flow in jitted sections; use existing named-jit and Equinox modules.

### Task Breakdown

1) Extract reusable generation backend
- [ ] Create `src/levanter/inference/service.py` with `GenerationService`:
  - [ ] Initialization: device mesh (`TrainerConfig`), tokenizer loading, `_load_model`, `PageTable.init`, `model.initial_cache`, `JitScheduler.init`, `DecodeState.init`, `Sampler`.
  - [ ] Public APIs:
    - [ ] `generate(prompt: str | list[str], options) -> GeneratedResult`
    - [ ] `stream_generate(...)-> AsyncIterator[StreamChunk]`
  - [ ] Internals mirror `sample_lm` flow: enqueue prompts, `run_generation_loop`, `extract_outputs`, stop-seq handling, temperature, seed.
  - [ ] Accept per-request options: `max_tokens`, `temperature`, `top_p` (stub ok), `stop` (list[str]), `n` (later), `logprobs` (later).
- [ ] Refactor `run_generation_loop` logic into a method usable from both sync and async paths; keep jitted parts unchanged.
- [ ] Unit tests targeting service with a tiny config (CPU) to decode 1–2 tokens.

2) HTTP server scaffold
- [ ] Add optional dependency group `serve`: FastAPI + Uvicorn (or Starlette). Optional: Pydantic v2.
- [ ] New module `src/levanter/serve/oai_server.py` exporting `app`.
- [ ] Endpoints:
  - [ ] `GET /healthz` returns 200 once model is loaded and warm.
  - [ ] `GET /v1/models` returns loaded model id(s).
  - [ ] `POST /v1/completions` for text completion (prompt-based).
  - [ ] `POST /v1/chat/completions` for chat.
- [ ] Authentication: accept `Authorization: Bearer <token>` if `LEVANTER_API_KEY` set; otherwise open.
- [ ] Streaming: if `stream=true`, return SSE with `data: {delta...}` chunks and `data: [DONE]` per OpenAI semantics.

3) Request/response schema mapping
- [ ] Define Pydantic models for requests/responses (subset of OpenAI spec):
  - [ ] Chat: `messages`, `model`, `max_tokens`, `temperature`, `top_p`, `stop`, `stream`, `user`.
  - [ ] Completions: `prompt`, `model`, same options.
- [ ] Response objects include `id`, `object`, `created`, `model`, `choices`, `usage`.
- [ ] Translate chat `messages` -> prompt via a simple, configurable chat template (default: Llama 3 style). Support `system` and `user` roles; ignore tools/functions initially.
- [ ] Compute `usage` based on tokenized prefix and generated tokens.

4) Scheduling, batching, and backpressure
- [ ] Async request queue that batches requests and feeds `JitScheduler`.
- [ ] Per-request state in `DecodeState` rows; enforce limits: `max_seq_len`, `pages_per_seq`, `page_size`, `max_new_tokens`.
- [ ] Backpressure: bound queue size and timeout; return 429/503 on saturation.
- [ ] Cancellation: detach sequence and `purge_queue_of_seq` on client disconnect.

5) Model loading, configuration, and CLI
- [ ] Support either `checkpoint_path` or `hf_checkpoint` (mutually exclusive), same as `SampleLmConfig`.
- [ ] Reuse `TrainerConfig` for device mesh and `mp` dtype; allow YAML config file.
- [ ] CLI `levanter-serve`:
  - [ ] Args: `--config`, `--hf-checkpoint`, `--checkpoint-path`, `--tokenizer`, `--port`, `--host`, `--max-batch-seqs`, `--max-queue-tokens`.
  - [ ] Warmup: perform one tiny decode to compile.

6) Observability
- [ ] Structured logging for lifecycle and request events (start, end, errors, latency, tps).
- [ ] Optional Prometheus `/metrics` exporter with request counts, latencies, tokens/sec, compile time.
- [ ] Error surfaces include truncated stack traces and 4xx/5xx mapping.

7) Testing
- [ ] Unit: schema validation, prompt templating, stop sequence application.
- [ ] Integration: spin up server with tiny CPU config; `POST /v1/chat/completions` generating <= 4 tokens.
- [ ] Streaming: verify incremental deltas and `[DONE]` terminator.
- [ ] Load smoke: concurrent requests (N=4–8) exercise batching path.

8) Packaging & deployment
- [ ] Add optional `docs/design/Dockerfile` for serving (uv + runtime). Build stage compiles deps; runtime is slim.
- [ ] Example Kubernetes manifest (`infra/cluster/`) with resource requests, liveness/readiness probes.
- [ ] Document TPU/GPU flags and any XLA env needed.

9) Documentation & examples
- [ ] `README` snippet with curl examples for both endpoints.
- [ ] OpenAI SDK example:
  - Python: set `OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")`.
  - Node: `new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "sk-local" })`.
- [ ] Note unsupported options initially (function calling, tools, `response_format`, images, vision).

10) Stretch/Follow-ups
- [ ] `n>1` parallel sampling and multi-choice responses.
- [ ] `logprobs` surface and token-level details.
- [ ] Presence/frequency penalties; repetition penalty.
- [ ] `/v1/embeddings` if/when model supports.
- [ ] Tracing (OpenTelemetry) and richer metrics.

### File/Code references
- `src/levanter/main/sample_lm.py`: `SampleLmConfig`, `_load_model`, `GenState`, `run_generation_loop`, `_one_round`, `extract_outputs`.
- `src/levanter/inference/jit_scheduler.py`: `JitScheduler`, `DecodeState`, `SeqDecodingParams`.
- `levanter.inference.page_table.PageTable`, `levanter.layers.attention.KvPageCache`.
- `levanter.layers.sampler.Sampler`.
- `levanter.models.llama.LlamaLMHeadModel` and `levanter.models.lm_model.LmHeadModel`.
- `levanter.compat.hf_checkpoints.HFCheckpointConverter`, `load_tokenizer`.
- `levanter.trainer.TrainerConfig` and `levanter.utils.jax_utils.use_cpu_device`.

### Acceptance criteria
- Can run: `levanter-serve --hf-checkpoint <org/model> --tokenizer <tok> --port 8000` and:
  - `GET /v1/models` returns model id.
  - `POST /v1/chat/completions` returns a valid OpenAI response; streaming works with SSE; `usage` is populated.
  - Concurrency: 4 parallel requests; batching occurs; no crashes; graceful shutdown.
