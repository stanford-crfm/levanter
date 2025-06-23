from typing import Mapping, Sequence

import equinox as eqx
import numpy as np

from haliax import Axis, NamedArray

from levanter.data._preprocessor import BatchProcessor
from levanter.models.dpo_example import DpoExample
from levanter.utils.hf_utils import num_cpus_used_by_tokenizer


class DpoProcessor(BatchProcessor[Mapping, DpoExample]):
    """
    Batch processor that converts raw DPO dicts to DpoExample instances with NamedArray fields.
    """

    def __init__(
        self,
        tokenizer,
        Prompt: Axis,
        Response: Axis,
    ):
        self.tokenizer = tokenizer
        self.Prompt = Prompt
        self.Response = Response

    def __call__(self, batch: Sequence[Mapping]) -> list[DpoExample]:
        return [DpoExample.from_dict(raw, self.tokenizer, self.Prompt, self.Response) for raw in batch]

    @property
    def output_exemplar(self) -> DpoExample:
        # exemplar with zeros for one example (no batch axis)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        import haliax as hax

        prompt = np.zeros((self.Prompt.size,), dtype=np.int32)
        chosen = np.zeros((self.Response.size,), dtype=np.int32)
        rejected = np.zeros((self.Response.size,), dtype=np.int32)
        return DpoExample(
            prompt_ids=hax.named(prompt, (self.Prompt,)),
            chosen_ids=hax.named(chosen, (self.Response,)),
            rejected_ids=hax.named(rejected, (self.Response,)),
            prompt_len=0,
            response_len=0,
        )

    @property
    def num_cpus(self) -> int:
        return num_cpus_used_by_tokenizer(self.tokenizer)

    @property
    def num_gpus(self) -> int:
        return 0

    @property
    def metadata(self) -> dict:
        return {}
