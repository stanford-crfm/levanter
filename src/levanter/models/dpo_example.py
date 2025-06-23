import equinox as eqx
import numpy as np

import haliax as hax
from haliax import Axis, NamedArray


class DpoExample(eqx.Module):
    prompt_ids: NamedArray  # axes=(Pos,)
    chosen_ids: NamedArray  # axes=(Response,)
    rejected_ids: NamedArray  # axes=(Response,)

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

        prompt = np.array(pad(raw["prompt_ids"], Pos.size), dtype=np.int32)
        chosen = np.array(pad(raw["chosen_ids"], Response.size), dtype=np.int32)
        rejected = np.array(pad(raw["rejected_ids"], Response.size), dtype=np.int32)

        # wrap in NamedArray without batch axis
        prompt_na = hax.named(prompt, (Pos,))
        chosen_na = hax.named(chosen, (Response,))
        rejected_na = hax.named(rejected, (Response,))

        return DpoExample(prompt_ids=prompt_na, chosen_ids=chosen_na, rejected_ids=rejected_na)
