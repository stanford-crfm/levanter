from dataclasses import dataclass
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from levanter.trainer import TrainerConfig
from pydantic import BaseModel, Field

from levanter.inference.service import GenerationService, GenerationOptions
from levanter.compat.hf_checkpoints import RepoRef
import levanter
import levanter.config


app = FastAPI(title="Levanter Minimal Completions Server")
_service: Optional[GenerationService] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    stop: Optional[str | list[str]] = None
    seed: Optional[int] = None
    n: int = Field(default=1, ge=1, le=16)  # Number of completions to generate


class Choice(BaseModel):
    index: int
    text: str
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ---- Chat Completions Schema ----
class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: Optional[str | list[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    n: int = Field(default=1, ge=1, le=16)


class ChatChoiceMessage(BaseModel):
    role: str
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatChoiceMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


@app.get("/healthz")
def healthz():
    ready = bool(_service and _service.ready())
    detail = getattr(_service, "last_error", None) if _service and not ready else None
    return {"status": "ok", "ready": ready, "detail": detail}


@app.get("/v1/models")
def list_models():
    """List available models."""
    if _service is None or not _service.ready():
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "object": "list",
        "data": [
            {
                "id": _service.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "levanter"
            }
        ]
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(req: CompletionRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")
    if _service is None or not _service.ready():
        raise HTTPException(status_code=503, detail=(getattr(_service, "last_error", None) if _service else "model not ready"))

    created = int(time.time())
    choices = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # Generate n completions concurrently
    generation_tasks = []
    for i in range(req.n):
        # Use different seeds for each generation if seed is provided
        seed = None
        if req.seed is not None:
            seed = req.seed + i  # Offset seed for each generation
        else:
            seed = created + i  # Use different seeds based on timestamp

        opts = GenerationOptions(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=(req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None)),
            seed=seed,
        )

        # Use async generation for better concurrency
        task = _service.generate_once_async(req.prompt, opts)
        generation_tasks.append((i, task))

    # Wait for all generations to complete
    for i, task in generation_tasks:
        result = await task
        choice = Choice(index=i, text=result.text, finish_reason=result.finish_reason)
        choices.append(choice)

        # Accumulate token counts (prompt tokens should be the same for all)
        total_prompt_tokens = result.prompt_tokens
        total_completion_tokens += result.completion_tokens

    usage = Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        object="text_completion",
        created=created,
        model=_service.model_id,
        choices=choices,
        usage=usage,
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    if _service is None or not _service.ready():
        raise HTTPException(status_code=503, detail=(getattr(_service, "last_error", None) if _service else "model not ready"))

    # Convert ChatMessage -> dicts expected by HF apply_chat_template
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        prompt = _service.apply_chat_template(messages, add_generation_prompt=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to render chat template: {e}")

    created = int(time.time())
    choices: list[ChatChoice] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    generation_tasks = []
    # Always include the chat turn delimiter as a stop sequence for chat
    # For now, hardcode to "<|user|>" (ideally sniff from tokenizer/template)
    base_stops: list[str] = ["<|user|>"]
    if req.stop is not None:
        if isinstance(req.stop, list):
            # Avoid duplicates and keep <|user|> first
            for s in req.stop:
                if s and s not in base_stops:
                    base_stops.append(s)
        else:
            if req.stop and req.stop not in base_stops:
                base_stops.append(req.stop)

    for i in range(req.n):
        seed = (req.seed + i) if req.seed is not None else (created + i)
        opts = GenerationOptions(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stop=base_stops,
            seed=seed,
        )
        generation_tasks.append((i, _service.generate_once_async(prompt, opts)))

    for i, task in generation_tasks:
        result = await task
        msg = ChatChoiceMessage(role="assistant", content=result.text)
        choices.append(ChatChoice(index=i, message=msg, finish_reason=result.finish_reason))
        total_prompt_tokens = result.prompt_tokens
        total_completion_tokens += result.completion_tokens

    usage = Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=created,
        model=_service.model_id,
        choices=choices,
        usage=usage,
    )


def main():
    @dataclass(frozen=True)
    class ServeConfig:
        trainer: TrainerConfig
        host: str = "0.0.0.0"
        port: int = 8000
        hf_checkpoint: Optional[RepoRef] = None
        checkpoint_path: Optional[str] = None
        tokenizer: Optional[str] = None
        seed: int = 0
        log_level: str = "info"
        access_log: bool = True

    def _run(cfg: ServeConfig):
        # Lazy import to avoid forcing uvicorn at import-time
        import uvicorn  # type: ignore

        # initialize jax/logging/tracker similar to other entrypoints
        levanter.initialize(cfg)

        global _service
        _service = GenerationService(
            hf_checkpoint=str(cfg.hf_checkpoint) if cfg.hf_checkpoint is not None else None,
            checkpoint_path=cfg.checkpoint_path,
            tokenizer=cfg.tokenizer,
            seed=cfg.seed,
            trainer=cfg.trainer,
        )

        # Warmup JIT compilation with a tiny generation
        print("Warming up JIT compilation...")
        try:
            warmup_result = _service.generate_once("Hello", GenerationOptions(max_tokens=1, temperature=0.7, seed=42))
            print(f"JIT warmup complete. Generated: {repr(warmup_result.text)}")
        except Exception as e:
            print(f"JIT warmup failed: {e}")
            # Continue anyway - the service might still work for some requests

        uvicorn.run(
            app,
            host=cfg.host,
            port=cfg.port,
            reload=False,
            workers=1,
            log_level=cfg.log_level,
            access_log=cfg.access_log,
        )

    levanter.config.main(_run)()


if __name__ == "__main__":
    main()
