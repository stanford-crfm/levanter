from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import List

from .sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Lifecycle state of a sequence."""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    """Tracks the state of a single generation request."""

    prompt_token_ids: List[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    seq_id: int = -1
    status: SequenceStatus = SequenceStatus.WAITING
    token_ids: List[int] = field(init=False)

    _counter = count()

    def __post_init__(self):
        if self.seq_id < 0:
            self.seq_id = next(self._counter)
        self.token_ids = list(self.prompt_token_ids)

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
