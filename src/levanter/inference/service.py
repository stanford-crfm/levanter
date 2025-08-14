from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationOptions:
    max_tokens: int = 16
    temperature: float = 0.7
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


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
    ):
        self._hf_checkpoint = hf_checkpoint
        self._checkpoint_path = checkpoint_path
        self._tokenizer = tokenizer
        self._seed = seed
        self._ready = True  # flip to True after real init

    def ready(self) -> bool:
        return self._ready

    @property
    def model_id(self) -> str:
        if self._hf_checkpoint is not None:
            return self._hf_checkpoint
        if self._checkpoint_path is not None:
            return self._checkpoint_path
        return "levanter-echo"

    def generate_once(self, prompt: str, options: GenerationOptions) -> str:
        # TODO: wire to real generation using sample_lm building blocks
        # For now, echo the prompt to validate server plumbing.
        return prompt
