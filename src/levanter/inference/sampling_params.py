from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Parameters controlling text generation sampling."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
