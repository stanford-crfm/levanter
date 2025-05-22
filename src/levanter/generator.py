import time
from dataclasses import dataclass
from typing import Literal, Tuple

import equinox as eqx
import jax
import numpy as np
from jax.random import PRNGKey
from jaxtyping import PRNGKeyArray
from tqdm.auto import tqdm
from typing_extensions import Self

from haliax import Axis, NamedArray
from haliax.nn import hax

from levanter.compat.hf_checkpoints import HFCompatConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.hf_utils import HfTokenizer


Vocab = hax.Axis("vocab", 128256)
Pos = hax.Axis("position", 64)
Batch = hax.Axis("batch", 1)


@dataclass
class SamplingOutput:
    """Struct for encapsulating the model output in a convenience wrapper."""

    output_tokens: list[int]  # TODO: Fix types
    num_tokens: int
    decoded_output: str
    total_time_taken: float  # for calculating tok/s


@dataclass(frozen=True)
class Prompt:
    """Struct for encapsulating the prompt and pretty print it."""

    system_instruction: str
    input_prompt: str
    user_format: str = " "

    def _to_string(self: Self) -> str:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. Write"
            " a response that appropriately completes the request in the provided format.\n\n### Instruction:\n"
            f" {self.system_instruction}\n### Input:\n {self.input_prompt}\n### Format: {self.user_format}###"
            " Response:"
        )

    def __repr__(self: Self) -> str:
        return f"{'-' * 10}\nModel Prompt:{self._to_string()}\n{'-' * 10}"


@dataclass(frozen=True)
class SamplingArgs:
    """Parent struct for encapsulating arguments provided to a sampler."""

    prompt: Prompt
    num_tokens: int = 128
    temperature: float = 0.5


@dataclass(frozen=True, kw_only=True)
class GreedySamplerArgs(SamplingArgs):
    """Struct to encapsulate arguments for the Greedy Sampler."""

    repetition_token_window: int = 4
    repetition_penalty: float = 0.8
    top_k: int = 19
    top_p: float = 0.9


C = LmConfig | HFCompatConfig


class Generator:
    hf_config: C
    tokenizer: HfTokenizer
    model: LmHeadModel[C]
    key: PRNGKeyArray

    def __init__(
        self,
        hf_config: C,
        tokenizer: HfTokenizer,
        model: LmHeadModel[C],
        key: PRNGKeyArray = PRNGKey(0),
    ) -> None:
        self.hf_config = hf_config
        self.tokenizer = tokenizer
        self.model = model
        self.key = key

    def greedy_sample(
        self: Self,
        sampling_args: GreedySamplerArgs,
        logits: NamedArray,
        last_tok_idx: NamedArray,  # NamedArray[int], size: (1,)
    ) -> NamedArray:
        """
        Samples the model using standard techniques:
        - Temperature scaling
        - Top-k sampling
        - Nucleus (top-p) sampling
        - Repetition penalty
        """

        penalty_vocab_mask = self._get_rep_penalty_mask(
            repetition_token_window=sampling_args.repetition_token_window,
            last_tok_idx=last_tok_idx,
            repetition_penalty=sampling_args.repetition_penalty,
        )

        # Extract the last token representation
        last_tok_logits = hax.take(logits / sampling_args.temperature, Pos, last_tok_idx)

        # Apply the repetition penalty
        last_tok_logits = last_tok_logits * penalty_vocab_mask

        # Top-k
        top_k_logits, axis, top_k_indices = self._get_top_k(sampling_args.top_k, last_tok_logits)

        # Top-p
        nucleus = self._get_top_p(top_k_logits, axis, sampling_args.top_p)

        token = hax.random.choice(
            key=self.key,
            shape=(),
            a=top_k_indices,
            p=nucleus["batch", 1],
            axis=axis,
        )

        return token

    def _tokenize(self: Self, input_str: str) -> Tuple[NamedArray, ...]:
        self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenized = self.tokenizer(
            input_str,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=Pos.size,
            pad_to_multiple_of=4,
            padding_side="right",
            add_special_tokens=True,
            return_attention_mask=True,
        )

        def to_named_arr(x) -> NamedArray:
            return hax.named(np.asarray(x), axis=(Batch, Pos))

        return to_named_arr(tokenized["input_ids"]), to_named_arr(tokenized["attention_mask"])

    def generate(
        self: Self,
        args: SamplingArgs,
        sampler: Literal["greedy_samping"] = "greedy_samping",
    ) -> SamplingOutput:
        prompt_ids, attn_mask = self._tokenize(args.prompt._to_string())

        match sampler:
            case "greedy_samping":
                assert isinstance(args, GreedySamplerArgs), "Wrong Sampler arguments provided."
                gen_sampler = self.greedy_sample

            case _:
                raise NotImplementedError(f"This sampler ({sampler}) has not been implemented yet. ")

        model = eqx.filter_jit(self.model)
        last_tok_idx = hax.sum(attn_mask) - 1  # attn_mask is guranteed binary

        def _scan_body(carry, _):
            prompt_ids, attn_mask, last_tok_idx = carry

            logits = model(prompt_ids, attn_mask)
            token = gen_sampler(args, logits, last_tok_idx)

            new_prompt_ids = prompt_ids.at["position", last_tok_idx + 1].set(token)
            new_attn_mask = attn_mask.at["position", last_tok_idx + 1].set(1)

            return (new_prompt_ids, new_attn_mask, last_tok_idx + 1), token

        initial_carry = (prompt_ids, attn_mask, last_tok_idx)

        start = time.time()

        final_carry, _ = jax.lax.scan(
            _scan_body, initial_carry, None, length=args.num_tokens  # No per-iteration sequence `xs` needed
        )

        final_carry[0].array.block_until_ready()

        print(f"Done scan. Taken: {time.time() - start}")

        start = time.time()

        for _ in tqdm(range(args.num_tokens)):
            logits = eqx.filter_jit(self.model)(prompt_ids, attn_mask)

            # print(
            #     "Greedy:",
            #     self.tokenizer.decode(
            #         hax.argmax(logits["position", last_tok_idx], Vocab).array
            #     ),
            # )

            token = gen_sampler(args, logits, last_tok_idx)

            prompt_ids = prompt_ids.at["position", last_tok_idx + 1].set(token)
            attn_mask = attn_mask.at["position", last_tok_idx + 1].set(1)
            last_tok_idx += 1

            # print(f"Decoded token: {self.tokenizer.decode(token.array)}")

        print(f"Done Naive. Taken: {time.time() - start}")

        model_output = self.tokenizer.decode(
            prompt_ids.array.squeeze(),
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )

        return SamplingOutput(
            prompt_ids.array[0].tolist(),
            args.num_tokens,
            model_output,
            total_time_taken=-1.0,
        )

    def _get_top_p(self, top_k_logits: NamedArray, axis: Axis, top_p: float) -> NamedArray:
        probs = hax.nn.softmax(top_k_logits, axis)
        cumulative_probs = hax.cumsum(probs, axis)

        nucleus = hax.where(cumulative_probs > top_p, 0.0, probs)  # clip where cumulative prob > `p`

        nucleus = nucleus / hax.sum(nucleus)  # ensure valid PDF

        return nucleus

    def _get_top_k(self, top_k: int, last_tok_logits: NamedArray) -> tuple[NamedArray, Axis, NamedArray]:

        K = hax.Axis("top_k", top_k)

        top_k_logits, top_k_indices = hax.top_k(last_tok_logits, Vocab, k=top_k, new_axis=K)
        return top_k_logits, K, top_k_indices

    def _get_rep_penalty_mask(
        self,
        repetition_token_window: int,
        last_tok_idx: NamedArray,
        repetition_penalty: float,
    ) -> NamedArray:
        penalty_vocab_mask = hax.ones(Vocab)

        assert Pos.size > repetition_token_window, "Token window to check for repetitions is < context length."

        R = hax.Axis("last_r", repetition_token_window)

        last_r_indices = (last_tok_idx - repetition_token_window) + hax.arange(R)

        # Mask of shape `(Vocab,)` where last `r` seen tokens are downweighted
        # Meant to be applied when decoding the logits of the last token.
        penalty_vocab_mask = hax.ones(Vocab).at["vocab", last_r_indices].set(repetition_penalty)

        return penalty_vocab_mask
