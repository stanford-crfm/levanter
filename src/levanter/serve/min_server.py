from __future__ import annotations

from dataclasses import dataclass
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
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
    return {"status": "ok", "ready": bool(_service and _service.ready())}


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")
    if _service is None or not _service.ready():
        raise HTTPException(status_code=503, detail="model not ready")
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
        host: str = "0.0.0.0"
        port: int = 8000
        hf_checkpoint: Optional[RepoRef] = None
        checkpoint_path: Optional[str] = None
        tokenizer: Optional[str] = None
        seed: int = 0

    def _run(cfg: ServeConfig):
        # Lazy import to avoid forcing uvicorn at import-time
        import uvicorn  # type: ignore

        global _service
        _service = GenerationService(
            hf_checkpoint=str(cfg.hf_checkpoint) if cfg.hf_checkpoint is not None else None,
            checkpoint_path=cfg.checkpoint_path,
            tokenizer=cfg.tokenizer,
            seed=cfg.seed,
        )

        uvicorn.run("levanter.serve.min_server:app", host=cfg.host, port=cfg.port, reload=False, workers=1)

    levanter.config.main(_run, ServeConfig)()


if __name__ == "__main__":
    main()
