# OAI-compatible Inference Server (Levanter)

> **Note**: For detailed implementation tasks and technical design, see [`docs/design/inference.md`](/docs/design/inference.md).

**Goal**: Expose Levanter models via an OpenAI-compatible HTTP API with streaming, batching, and scheduler-backed decoding. Reuse `src/levanter/main/sample_lm.py` and dependencies: `JitScheduler`, `DecodeState`, `PageTable`, `KvPageCache`, `Sampler`, tokenizer/HF loader, and `TrainerConfig` device mesh setup.

## Current Milestone: Minimal single-turn completions (FastAPI)

**Scope**: Implement only `POST /v1/completions` (no chat, no streaming, batch size = 1). Single request -> single output text. Use tokenizer + model via a simplified wrapper of `sample_lm` logic (prefill + decode loop), with `max_tokens`, `temperature`, and optional `stop` (string or list of strings). Return OpenAI completion response shape.

**Out of scope**: SSE streaming, batching >1, usage tokens accounting beyond basic counts, top_p/top_k penalties, function/tool calls.

### Progress Tracking

- [x] **Server scaffold**
  - [x] Add optional dependency group `serve` (fastapi + uvicorn).
  - [x] Create `src/levanter/serve/min_server.py` exposing `app` and a `main()` to run uvicorn.
  - [x] Endpoint: `POST /v1/completions` with minimal OpenAI-compatible schema.
  - [x] Health: `GET /healthz` returns 200 when model is ready.

- [x] **Minimal model wrapper**
  - [x] `GenerationService` (single-seq) that loads tokenizer + model via `_load_model` and sets up `PageTable`, `KvPageCache`, `JitScheduler`, `DecodeState`, `Sampler`.
  - [x] `generate_once(prompt, max_tokens, temperature, stop) -> text` using the same enqueue + `run_generation_loop` + `extract_outputs` flow as `sample_lm`, but limited to a single sequence and a blocking loop until finish or `max_tokens`.
  - [x] Convert `stop` strings to token sequences with tokenizer and pass via `SeqDecodingParams`.

- [x] **Request/response models** (subset of OpenAI)
  - [x] Request: `{ model: str, prompt: str, max_tokens?: int, temperature?: float, stop?: str | string[], seed?: int }`
  - [x] Response: `{ id, object: "text_completion", created, model, choices: [{ index, text, finish_reason }], usage: { prompt_tokens, completion_tokens, total_tokens } }`

- [x] **CLI**
  - [x] `python -m levanter.serve.min_server --host 0.0.0.0 --port 8000 --hf_checkpoint <org/model> --tokenizer <org/model>`
  - [x] Warmup one-token decode on startup to trigger JIT.

- [x] **Acceptance criteria**
  - [x] `curl -X POST http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{"model":"<id>","prompt":"Hello","max_tokens":8}'` returns a valid response with non-empty `choices[0].text`.
  - [ ] End-to-end runs on CPU with a tiny HF model; no crashes; initial JIT compile acknowledged in logs.

### Next Steps

1. **Warmup JIT compilation** on startup
2. **Fix usage token counting** to use actual tokenizer instead of `len(text.split())`
3. **Test end-to-end** with a tiny HF model on CPU
4. **Add `/v1/models` endpoint** to return model information

## Future Milestones

### Milestone: Chat Completions
- [ ] Implement `POST /v1/chat/completions` endpoint
- [ ] Add chat message templating (Llama 3 style)
- [ ] Support streaming responses with SSE

### Milestone: Batching & Concurrency
- [ ] Implement request queuing and batching
- [ ] Add concurrent request handling
- [ ] Implement graceful backpressure

### Milestone: Production Features
- [ ] Add authentication (optional API key)
- [ ] Implement proper metrics and observability
- [ ] Add graceful shutdown handling
- [ ] Create Docker container and deployment examples

## Notes

This milestone deliberately avoids concurrency and streaming to minimize moving parts while validating the integration of tokenizer, model, scheduler, and response schema.

For detailed implementation tasks and technical architecture, see [`docs/design/inference.md`](/docs/design/inference.md).
