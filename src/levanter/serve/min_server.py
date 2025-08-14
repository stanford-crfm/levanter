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


@app.get("/healthz")
def healthz():
    ready = bool(_service and _service.ready())
    detail = getattr(_service, "last_error", None) if _service and not ready else None
    return {"status": "ok", "ready": ready, "detail": detail}


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")
    if _service is None or not _service.ready():
        raise HTTPException(status_code=503, detail=(getattr(_service, "last_error", None) if _service else "model not ready"))
    created = int(time.time())
    opts = GenerationOptions(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=(req.stop if isinstance(req.stop, list) else ([req.stop] if req.stop else None)),
        seed=req.seed if req.seed is not None else created,
    )
    text = _service.generate_once(req.prompt, opts)
    choice = Choice(index=0, text=text, finish_reason="stop")
    # Rough usage until wired to tokenizer counts
    prompt_tokens = len(req.prompt.split())
    completion_tokens = len(text.split()) - prompt_tokens if text.startswith(req.prompt) else len(text.split())
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=max(completion_tokens, 0), total_tokens=prompt_tokens + max(completion_tokens, 0))
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        object="text_completion",
        created=created,
        model=_service.model_id,
        choices=[choice],
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
            warmup_text = _service.generate_once("Hello", GenerationOptions(max_tokens=1, temperature=0.7, seed=42))
            print(f"JIT warmup complete. Generated: {repr(warmup_text)}")
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
