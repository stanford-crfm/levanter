import equinox as eqx
import numpy as np

import haliax as hax
from haliax import Axis, NamedArray


class DpoExample(eqx.Module):
    prompt_ids: NamedArray  # axes=(Pos,)
    chosen_ids: NamedArray  # axes=(Response,)
    rejected_ids: NamedArray  # axes=(Response,)
    prompt_len: int = 0  # number of prompt tokens before padding
    response_len: int = 0  # number of response tokens before padding

    @staticmethod
    def from_dict(
        raw: dict,
        tokenizer,
        Pos: Axis,
        Response: Axis,
    ) -> "DpoExample":
        """
        Build a DpoExample from raw token id lists in raw dict.
        Pads/truncates to Pos.size and Response.size, wraps in NamedArray without batch axis.
        """
        pad_id = getattr(tokenizer, "pad_token_id", 0)

        def pad(seq, target_len):
            # Unwrap NamedArray to raw array if needed
            if hasattr(seq, "array"):
                raw = seq.array
            else:
                raw = seq
            # Convert to Python list
            if isinstance(raw, np.ndarray):
                lst = raw.tolist()
            else:
                lst = list(raw)
            # Truncate or pad to target_len
            out = lst[:target_len]
            if len(out) < target_len:
                out += [pad_id] * (target_len - len(out))
            return out

        raw_prompt = raw["prompt_ids"]
        raw_chosen = raw["chosen_ids"]
        raw_rejected = raw["rejected_ids"]

        prompt_len = int(raw.get("prompt_len", min(len(raw_prompt), Pos.size)))
        chosen_len = min(len(raw_chosen), Response.size)
        rejected_len = min(len(raw_rejected), Response.size)

        response_len = int(raw.get("response_len", min(chosen_len, rejected_len)))

        prompt = np.array(pad(raw_prompt, Pos.size), dtype=np.int32)
        chosen = np.array(pad(raw_chosen, Response.size), dtype=np.int32)
        rejected = np.array(pad(raw_rejected, Response.size), dtype=np.int32)

        # wrap in NamedArray without batch axis
        prompt_na = hax.named(prompt, (Pos,))
        chosen_na = hax.named(chosen, (Response,))
        rejected_na = hax.named(rejected, (Response,))

        return DpoExample(
            prompt_ids=prompt_na,
            chosen_ids=chosen_na,
            rejected_ids=rejected_na,
            prompt_len=prompt_len,
            response_len=response_len,
        )
