# Code for running https://github.com/EleutherAI/lm-evaluation-harness inside Levanter runs
# References:
# https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py
# https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/TPU_cluster.py#L6
import dataclasses
import functools
import json
import logging
import tempfile
import typing
from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax

import levanter.tracker
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.models.gpt2 import Gpt2Config
from levanter.models.loss import next_token_loss


try:
    from lm_eval import evaluator
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
except ImportError:
    LM = object
    Instance = object
    evaluator = object

from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import round_axis_for_partitioning

import levanter.config
from levanter.checkpoint import load_checkpoint
from levanter.data import AsyncDataset, DataLoader
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import StepInfo, TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


class EvalDataset(AsyncDataset[LmExample]):
    def __init__(self, Pos, tokenizer, examples: list[Instance]):
        super().__init__()
        self.examples = examples
        self.Pos = Pos
        self.tokenizer = tokenizer

    async def async_len(self) -> int:
        return len(self.examples)

    async def final_length_is_known(self) -> bool:
        return True

    def is_finite(self) -> bool:
        return True

    async def current_len(self) -> Optional[int]:
        return len(self.examples)

    async def get_batch(self, indices: Sequence[int]) -> List[LmExample]:
        out = []
        pad_token_id = self.tokenizer.pad_token_id

        reqs = [(self.examples[i].args[0], self.examples[i].args[1]) for i in indices]

        for context, completion in reqs:
            whole_enc = self.tokenizer(context + completion)
            context_enc = self.tokenizer(context)

            context_enc_len = len(context_enc["input_ids"])

            tokens, length = self._truncate_or_pad(whole_enc, context_enc_len)
            example = _jit_create_example(self.Pos, tokens, length, pad_token_id)

            out.append(example)

        return out

    def _truncate_or_pad(self, encoded, prompt_length):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        ex_pad = self.tokenizer.pad(
            encoded,
            padding="max_length",
            max_length=self.Pos.size,
            return_tensors="np",
        )

        truncated = ex_pad["input_ids"][-self.Pos.size :]
        # if we truncated the prompt, we need to adjust the prompt length
        if len(truncated) < len(encoded):
            prompt_length -= len(encoded) - len(truncated)
            if prompt_length < 0:
                prompt_length = 0
                logger.warning("Prompt length is negative after truncation. Setting to 0.")

        return truncated, prompt_length


class LevanterHarnessLM(LM):
    def __init__(self, EvalBatch: hax.Axis, EvalPos: hax.Axis, model: LmHeadModel, axis_resources, tokenizer):
        super().__init__()
        self.EvalBatch = EvalBatch
        self.EvalPos = EvalPos
        self.model = model
        self.axis_resources = axis_resources
        self.tokenizer = tokenizer

        def _eval_loglikelihood(model: LmHeadModel, example: LmExample):
            logits = model(example.tokens, attn_mask=example.attn_mask)
            logits = logits.astype(jnp.float32)
            Pos = logits.resolve_axis(self.EvalPos.name)

            loss = next_token_loss(
                Pos=Pos,
                Vocab=model.Vocab,
                logits=logits,
                true_ids=example.tokens,
                loss_mask=example.loss_mask,
                reduction=hax.sum,
                reduction_axis=Pos,
            )

            not_last_loss_mask = 1 - hax.nn.one_hot(-1, Pos, dtype=bool)
            pred_targets = hax.argmax(logits, axis=model.Vocab)
            targets = hax.roll(example.tokens, -1, axis=Pos)
            freebie = hax.logical_not(example.loss_mask * not_last_loss_mask)
            correct = hax.all(hax.equal(pred_targets, targets) + freebie, axis=Pos)

            return -loss, correct

        # no sharded outputs
        self._jit_loglikelihood = hax.named_jit(
            _eval_loglikelihood, axis_resources=axis_resources, out_axis_resources={}
        )

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        """
        # pad requests to be a multiple of the batch size
        initial_length = len(requests)
        dummy_instance = dataclasses.replace(requests[0], arguments=("hello", " there"), idx=len(requests))
        requests = requests + [dummy_instance] * (self.EvalBatch.size - len(requests) % self.EvalBatch.size)
        assert len(requests) % self.EvalBatch.size == 0
        dataset = EvalDataset(self.EvalPos, self.tokenizer, requests)

        mesh = haliax.partitioning._get_mesh()

        loader = DataLoader(
            self.EvalBatch, dataset, max_buffered_batches=1024, mesh=mesh, axis_resources=self.axis_resources
        )

        result: list[tuple[float, bool]] = []
        for batch in tqdm(loader, desc="Loglikelihood", unit="ba"):
            out_lls, out_correct = self._jit_loglikelihood(self.model, batch)
            result.extend((ll.item(), correct.item()) for ll, correct in zip(out_lls.array, out_correct.array))

        assert len(result) >= initial_length
        # skip padding
        result = result[:initial_length]

        return result

    def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
        raise NotImplementedError()

    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError()


@functools.partial(jax.jit, static_argnums=(0, 3))
def _jit_create_example(Pos, tokens, prompt_len, pad_token_id):
    tokens = hax.named(tokens, Pos)
    return LmExample.from_prompt_and_completion(Pos, tokens, prompt_len, ignore_id=pad_token_id)


@dataclass(frozen=True)
class TaskConfig:
    """
    This is a dataclass that represents the configuration for a task in the LM Eval Harness. It is used to specify
    the configuration for a task in the LM Eval Harness, and is used to generate the task dictionary that the LM Eval
    Harness expects.

    nb that LM Eval Harness has its own TaskConfig, but its defaults are not the same as just passing in
    a dict, and we want the behavior of passing in a dict.

    See Also:
        [LM Eval Harness TaskConfig](https://github.com/EleutherAI/lm-evaluation-harness/blob/0ef7548d7c3f01108e7c12900a5e5eb4b4a668f7/lm_eval/api/task.py#L55)
    """

    task: str
    task_alias: str | None = None
    num_fewshot: int | None = None

    use_prompt: str | None = None
    description: str | None = None
    target_delimiter: str | None = None
    fewshot_delimiter: str | None = None

    def to_dict(self):
        base_dict = dataclasses.asdict(self)
        return {k: v for k, v in base_dict.items() if v is not None}


@dataclass(frozen=True)
class LmEvalHarnessConfig:
    task_spec: list[TaskConfig | str] | None = None
    max_examples: int | None = None
    max_eval_length: int | None = None
    log_samples: bool = False

    def task_spec_or_default(self) -> list[str | dict]:
        if self.task_spec is None:
            return ["hellaswag", "piqa"]
        return [task.to_dict() if isinstance(task, TaskConfig) else task for task in self.task_spec]

    def to_task_dict(self) -> dict:
        import lm_eval.tasks as tasks

        manager = tasks.TaskManager()
        # we need to do it this way b/c i can't figure out how to run e.g. hellaswag 0 shot and 10 shot in a single run
        this_tasks = {}
        for task in self.task_spec_or_default():
            try:
                if isinstance(task, str):
                    this_tasks.update(tasks.get_task_dict(task, manager))
                else:
                    our_name = task.get("task_alias", task["task"]) if isinstance(task, dict) else task
                    our_name = our_name.replace(" ", "_")
                    task_dict = tasks.get_task_dict([task], manager)
                    this_task = task_dict.popitem()[1]
                    # hacky, but this allows us to run multiple instances of the same task with different fewshot settings
                    this_task.config.task = our_name
                    this_tasks[our_name] = this_task
            except Exception:
                logger.exception(f"Failed to load task {task}")
                raise ValueError(f"Failed to load task {task}")
        return this_tasks


@dataclass(frozen=True)
class EvalHarnessMainConfig:
    tokenizer: str
    checkpoint_path: str
    checkpoint_is_hf: bool = False
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    model: LmConfig = dataclasses.field(default_factory=Gpt2Config)

    eval_harness: LmEvalHarnessConfig = dataclasses.field(default_factory=LmEvalHarnessConfig)

    @property
    def EvalBatch(self):
        return self.trainer.EvalBatch

    @cached_property
    def the_tokenizer(self):
        return load_tokenizer(self.tokenizer)


def run_lm_eval_harness(
    config: LmEvalHarnessConfig,
    model,
    tokenizer,
    EvalBatch,
    axis_resources,
) -> dict:
    # tasks_to_run = tasks.get_task_dict(config.task_spec_or_default(), tasks.TaskManager())
    tasks_to_run = config.to_task_dict()

    outputs = _actually_run_eval_harness(config, model, tasks_to_run, tokenizer, EvalBatch, axis_resources)

    return outputs


def _actually_run_eval_harness(config: LmEvalHarnessConfig, model, tasks_to_run, tokenizer, EvalBatch, axis_resources):
    max_examples = config.max_examples
    max_eval_length = config.max_eval_length

    EvalPos = model.Pos if max_eval_length is None else model.Pos.resize(max_eval_length)
    harness = LevanterHarnessLM(EvalBatch, EvalPos, model, axis_resources, tokenizer)
    outputs = evaluator.evaluate(harness, tasks_to_run, limit=max_examples, log_samples=config.log_samples)
    return outputs


def run_eval_harness_main(config: EvalHarnessMainConfig):
    config.trainer.initialize()
    tokenizer = config.the_tokenizer

    compute_axis_mapping = config.trainer.compute_axis_mapping
    parameter_axis_mapping = config.trainer.parameter_axis_mapping

    with config.trainer.device_mesh, hax.axis_mapping(parameter_axis_mapping):
        key = jax.random.PRNGKey(0)

        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), compute_axis_mapping)
        if vocab_size != Vocab.size:
            logger.info(f"Rounding vocab size from {vocab_size} to {Vocab.size} for partitioning")

        model: LmHeadModel

        # initialize the model
        if config.checkpoint_is_hf:
            model_config = config.model
            converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=config.checkpoint_path, tokenizer=tokenizer)
            model = converter.load_pretrained(
                model_config.model_type, ref=config.checkpoint_path, dtype=config.trainer.mp.compute_dtype  # type: ignore
            )
        else:
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
                model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = hax.shard(model, parameter_axis_mapping)

        model = typing.cast(LmHeadModel, inference_mode(model, True))

        logger.info("Running LM eval harness....")
        outputs = run_lm_eval_harness(
            config.eval_harness,
            model,
            tokenizer,
            config.EvalBatch,
            axis_resources=compute_axis_mapping,
        )

        logger.info("Finished running LM eval harness")
        # log the results as json
        with open("lm_eval_results.json", "w") as f:
            json.dump(outputs, f, indent=2)

        # also write to stdout
        if jax.process_index() == 0:
            print(json.dumps(outputs, indent=2), flush=True)

        # also log the results
        levanter.tracker.current_tracker().log_artifact("lm_eval_results.json")
        log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())

        return outputs


def log_report_to_tracker(prefix: str, report: dict, tracker: Optional[levanter.tracker.Tracker] = None):
    if tracker is None:
        tracker = levanter.tracker.current_tracker()

    to_log = {}
    for task_name, task_results in report["results"].items():
        for metric_name, metric_value in task_results.items():
            if metric_name.endswith(",none"):
                metric_name = metric_name[:-5]

            if isinstance(metric_value, float | int):
                to_log[f"{prefix}/{task_name}/{metric_name}"] = metric_value

    tracker.log(to_log, step=None)


def lm_eval_harness(config: LmEvalHarnessConfig, tokenizer, EvalBatch, axis_resources):
    tasks_to_run = config.to_task_dict()

    def lm_eval_harness(step: StepInfo, force=False):
        # if step.step == 0 and not force:
        #     return  # don't run eval on the first step

        print(config.task_spec_or_default())

        model = inference_mode(step.model, True)
        outputs = _actually_run_eval_harness(
            config,
            model,
            tasks_to_run,
            tokenizer,
            EvalBatch,
            axis_resources,
        )

        if jax.process_index() == 0:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
                import json

                json.dump(outputs, f)
                levanter.tracker.current_tracker().log_artifact(
                    f.name, name=f"lm_eval_output.{step.step}", type="lm_eval_output"
                )

            log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())

    return lm_eval_harness


if __name__ == "__main__":
    levanter.config.main(run_eval_harness_main)()
    print("Done", flush=True)
