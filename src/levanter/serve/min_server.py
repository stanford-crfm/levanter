from __future__ import annotations

import argparse
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Levanter Minimal Completions Server")


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    stop: Optional[str | list[str]] = None


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
    return {"status": "ok"}


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(req: CompletionRequest):
    # Stub implementation: echo back the prompt. Will be wired to GenerationService.
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")
    created = int(time.time())
    text = req.prompt  # placeholder
    choice = Choice(index=0, text=text, finish_reason="length" if req.max_tokens == 0 else "stop")
    usage = Usage(prompt_tokens=len(req.prompt.split()), completion_tokens=0, total_tokens=len(req.prompt.split()))
    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:12]}",
        object="text_completion",
        created=created,
        model=req.model,
        choices=[choice],
        usage=usage,
    )


def main():
    parser = argparse.ArgumentParser(description="Levanter Minimal Completions Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Lazy import to avoid forcing uvicorn at import-time
    import uvicorn  # type: ignore

    uvicorn.run("levanter.serve.min_server:app", host=args.host, port=args.port, reload=False, workers=1)


if __name__ == "__main__":
    main()
