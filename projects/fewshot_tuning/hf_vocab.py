from typing import List, Optional

import seqio
import tensorflow as tf
from transformers import PreTrainedTokenizerBase


class HfVocabulary(seqio.Vocabulary):
    """A Vocabulary implementation that uses a HuggingFace tokenizer."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def eos_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self) -> int:
        # GPT-2 doesn't have a pad token, so we use eos if not present
        id = self.tokenizer.pad_token_id
        if id is None or id < 0:
            return self.tokenizer.eos_token_id

        return id

    def unk_id(self) -> Optional[int]:
        return self.tokenizer.unk_token_id

    @property
    def extra_ids(self) -> int:
        # hf tokenizer manages its own special tokens
        return 0

    def __len__(self):
        return len(self.tokenizer)

    def __eq__(self, other):
        return isinstance(other, HfVocabulary) and self.tokenizer == other.tokenizer

    @property
    def _base_vocab_size(self) -> int:
        return len(self.tokenizer)

    def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
        def do_encode_tf(s: tf.Tensor):
            s = s.numpy().decode("utf-8")
            toks = self.tokenizer.encode(s, add_special_tokens=False, return_tensors="tf")
            assert len(toks.shape) == 2
            return tf.cast(toks[0], tf.int32)

        try:
            do_encode_tf(s)
        except AttributeError:
            return tf.py_function(func=do_encode_tf, inp=[s], Tout=tf.int32)


    def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
        return self.tokenizer.decode(ids)
