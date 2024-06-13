# Code for running https://github.com/EleutherAI/lm-evaluation-harness inside Levanter runs
# References:
# https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py
# https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/TPU_cluster.py#L6
import dataclasses
import json
import logging
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import transformers
from jax.experimental.multihost_utils import broadcast_one_to_all


try:
    from lm_eval import evaluator, tasks
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
except ImportError:
    LM = object
    Instance = object
    evaluator = object
    tasks = object

from tqdm import tqdm

import haliax as hax
from haliax.nn import cross_entropy_loss
from haliax.partitioning import round_axis_for_partitioning

import levanter.config
from levanter.checkpoint import load_checkpoint
from levanter.data import batched
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import local_cpu_mesh, stack_tree, use_cpu_device
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


# Ok this is a bit complicated to do because it's distributed systems and that's always hard.
# The idea is that we want to pass an LM adaptor to the harness, and then the harness will call the LM adaptor
# with a request, which we'll format, shard, and send to the model. The model will then return the result to the harness
# which will then return the result to the user.

# As we so often do, we will coordinate execution through JAX itself.

# Process 0 will:
# - Pass an adaptor to the eval harness
# - The eval harness will call the adaptor with a request
# - When a request comes in, it will call broadcast_one_to_all with a (REQUEST_TYPE, request) to send the request
# - It then invokes the model with the request and returns the result to the eval harness
# - When finished, it will call broadcast_one_to_all with a (FINISHED_TYPE, result) to send the result

# Process 1..n will:
# - Wait for a (REQUEST_TYPE, request) broadcast
# - if FINISHED_TYPE, break
# - Invoke the model with the request
# - loop


class _RequestType:
    LOG_LIKELIHOOD = 0
    GENERATE_UNTIL = 1
    LOG_LIKELIHOOD_ROLLING = 2
    FINISHED = 3


class _InternalLMAdapter:
    def __init__(self, EvalBatch: hax.Axis, model: LmHeadModel, axis_resources):
        self.EvalBatch = EvalBatch
        self.model = model
        self.axis_resources = axis_resources

        self.dummy_example: LmExample = hax.named_jit(lambda: LmExample.causal(
            tokens=hax.zeros(model.Pos, dtype=jnp.int32),
        ))()

        def _eval_loglikelihood(model: LmHeadModel, example: LmExample):
            logits = model(example.tokens)

            targets = hax.roll(example.tokens, -1, axis=model.Pos.name)
            target_y = hax.nn.one_hot(targets, model.Vocab, dtype=logits.dtype)
            loss = cross_entropy_loss(logits, model.Vocab, target_y, where=example.loss_mask, reduction_axis=model.Pos)
            # to tell if we got the right answer, we want to check that argmax(logits) == tokens wherever loss_mask is 1
            pred_targets = hax.argmax(logits, axis=model.Vocab)
            correct = hax.all(hax.equal(pred_targets, targets) | hax.logical_not(example.loss_mask), axis=model.Pos)

            return loss, correct

        # no sharded outputs
        self._jit_loglikelihood = hax.named_jit(
            _eval_loglikelihood, axis_resources=axis_resources, out_axis_resources={}
        )

    def loglikelihood(self, examples: LmExample) -> tuple[hax.NamedArray, hax.NamedArray]:
        return self._jit_loglikelihood(self.model, examples)

    def do_request(self, request_type, example):
        if request_type == _RequestType.LOG_LIKELIHOOD:
            return self.loglikelihood(example)
        elif request_type == _RequestType.GENERATE_UNTIL:
            raise NotImplementedError()
        elif request_type == _RequestType.LOG_LIKELIHOOD_ROLLING:
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid request type {request_type}")

    def worker_loop(self):
        dummy_example = self.dummy_example
        while True:
            logger.info("Waiting for request")
            request_type, request = broadcast_one_to_all((_RequestType.FINISHED, dummy_example), is_source=False)

            if request_type == _RequestType.FINISHED:
                break

            logger.info("Doing request")
            self.do_request(request_type, request)

    def dispatch(self, request_type, request):
        print(self.dummy_example, request)
        broadcast_one_to_all((request_type, request), is_source=True)
        return self.do_request(request_type, request)

    def finish(self):
        assert jax.process_index() == 0
        broadcast_one_to_all((_RequestType.FINISHED, self.dummy_example), is_source=True)


class LevanterHarnessLM(LM):
    def __init__(self, adaptor: _InternalLMAdapter, tokenizer):
        super().__init__()
        self.adaptor = adaptor
        self.tokenizer = tokenizer

        def _stack_batch(examples):
            return stack_tree(self.adaptor.EvalBatch, examples, pad_to_batch_size=True)

        self._stack_batch_jit = hax.named_jit(_stack_batch)

    def _stack_batch(self, examples):
        with use_cpu_device():
            return self._stack_batch_jit(examples)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        Args:
            requests:

        Returns:

        """

        Pos = self.adaptor.model.Pos

        contexts = self.tokenizer([req.args[0] for req in requests])["input_ids"]
        completions = self.tokenizer([req.args[1] for req in requests])["input_ids"]

        examples: list[LmExample] = []
        logger.info("Creating examples")
        with local_cpu_mesh():
            for context, completion in zip(contexts, completions):
                context, completion = self._truncate_or_pad(context, completion)
                context, completion = hax.named(context, Pos.name), hax.named(completion, Pos.name)
                example = jax.jit(lambda: LmExample.from_prompt_and_completion(
                    context, completion, ignore_id=self.tokenizer.pad_token_id
                ))()
                examples.append(example)
        logger.info("Created examples")

        result: list[tuple[float, bool]] = []
        for batch in batched(tqdm(examples, desc="examples", leave=False), self.adaptor.EvalBatch.size):
            logger.info("Processing batch")
            with local_cpu_mesh():
                batch_example = self._stack_batch(batch)
            out_lls, out_correct = self.adaptor.dispatch(_RequestType.LOG_LIKELIHOOD, batch_example)
            result.extend((ll.item(), correct.item()) for ll, correct in zip(out_lls.array, out_correct.array))

        # skip padding
        result = result[: len(examples)]

        # print(contexts)
        # print(completions)

        return result

    def _truncate_or_pad(self, context, completion):
        max_len = self.adaptor.model.Pos.size
        if len(completion) > max_len:
            warnings.warn(f"Completion is longer than max length {max_len}. Truncating.")
            completion = completion[:max_len]
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        if len(context) + len(completion) > max_len:
            context = context[-(max_len - len(completion)) :]
        else:
            # right pad with padding token
            context = context + [pad_token_id] * (max_len - len(context) - len(completion))

        return context, completion


    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError()


def run_lm_eval_harness(model, task_spec: list[str], tokenizer, EvalBatch, axis_resources, max_examples=None) -> dict:
    adapter = _InternalLMAdapter(EvalBatch, model, axis_resources)
    harness = LevanterHarnessLM(adapter, tokenizer)
    tasks_to_run = tasks.get_task_dict(task_spec)
    if jax.process_index() == 0:
        try:
            outputs = evaluator.evaluate(harness, tasks_to_run, limit=max_examples)
        finally:
            adapter.finish()
    else:
        adapter.worker_loop()
        outputs = {}

    return outputs


@dataclass(frozen=True)
class LmEvalHarnessConfig:
    task_spec: Optional[list[str]] = None
    max_examples: Optional[int] = None

    def task_spec_or_default(self):
        return self.task_spec or [
            # "lambada",
            # "piqa",
            "hellaswag",
            # "winogrande",
            # "mathqa",
            # "pubmedqa",
            # "boolq",
            # "cb",
            # "copa",
            # "multirc",
            # "record",
            # "wic",
            # "wsc",
        ]


@dataclass(frozen=True)
class EvalHarnessConfig:
    tokenizer: str
    checkpoint_path: str
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    model: LmConfig = dataclasses.field(default_factory=LmConfig)

    eval_harness: LmEvalHarnessConfig = dataclasses.field(default_factory=LmEvalHarnessConfig)

    @property
    def EvalBatch(self):
        return self.trainer.EvalBatch

    @cached_property
    def the_tokenizer(self):
        return transformers.AutoTokenizer.from_pretrained(self.tokenizer)


def run_eval_harness_main(config: EvalHarnessConfig):
    config.trainer.initialize()
    tokenizer = config.the_tokenizer

    task_spec = config.eval_harness.task_spec_or_default()
    max_examples = config.eval_harness.max_examples

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        # initialize the model
        if config.checkpoint_path is not None:
            # initialize the model
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
                model = inference_mode(model, True)

            model = hax.shard_with_axis_mapping(model, parameter_axis_mapping)

            logger.info("Running LM eval harness....")
            outputs = run_lm_eval_harness(
                model,
                task_spec,
                tokenizer,
                config.EvalBatch,
                axis_resources=compute_axis_mapping,
                max_examples=max_examples,
            )

            logger.info("Finished running LM eval harness")
            # log the results as json
            with open("lm_eval_results.json", "w") as f:

                json.dump(outputs, f, indent=2)
        else:
            raise ValueError("No checkpoint path provided")


if __name__ == "__main__":
    levanter.config.main(run_eval_harness_main)()
