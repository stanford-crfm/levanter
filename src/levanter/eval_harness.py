"""
This module contains code for running the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
inside Levanter runs. The EleutherAI LM Evaluation Harness is a tool for evaluating language models on a variety of tasks.

The [run_lm_eval_harness][] function runs the EleutherAI LM Evaluation Harness on a given model and tasks, and returns the
results.

It can also be used as a callback, via the [lm_eval_harness][] function.

Note that Levanter does not support generation (use VLLM or something) and the [generate_until][] method is not implemented.
So we only support tasks that work with loglikelihood, which is most(?) of them.

References:

* https://github.com/kingoflolz/mesh-transformer-jax/blob/master/eval_harness.py
* https://github.com/kingoflolz/mesh-transformer-jax/blob/f8315e3003033b23f21d78361b288953064e0e76/mesh_transformer/TPU_cluster.py#L6

"""
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
import numpy as np

import haliax
from haliax import NamedArray

import levanter.tracker
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.models.gpt2 import Gpt2Config
from levanter.models.loss import next_token_loss
from levanter.utils.hf_utils import HfTokenizer


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
from haliax.partitioning import ResourceMapping, round_axis_for_partitioning

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

        # lm-harness specs that args are (context, completion)
        reqs = [(self.examples[i].args[0], self.examples[i].args[1]) for i in indices]

        for context, completion in reqs:
            # it's kinda annoying we run tokenization twice, but it's the easiest way to get the prompt length
            # CF: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py#L354
            whole_enc = self.tokenizer(context + completion)
            context_enc = self.tokenizer(context)

            context_enc_len = len(context_enc["input_ids"])

            tokens, length = self._truncate_or_pad(whole_enc, context_enc_len)
            example = _jit_create_example(self.Pos, tokens, length, pad_token_id)

            out.append(example)

        return out

    def _truncate_or_pad(self, encoded: list[int], prompt_length: int):
        """
        Truncate or pad the encoded sequence to the maximum length of the model.
        Truncates from the beginning of the sequence, so that the completion is preserved.

        Returns:
            Truncated or padded sequence and the prompt length. The prompt length can be shorter than the original
            length if the input was truncated.
        """
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

        def _eval_loglikelihood(model: LmHeadModel, example: LmExample) -> tuple[NamedArray, NamedArray]:
            """
            Returns:
                - loss: The negative log-likelihood of the completion.
                - correct: Whether the completion is correct
            """
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
            # "freebie" is the positions we don't need to predict (prompt or final token's next token)
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
        dataset = self._pad_dataset_to_batch_size(requests)

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

    def _pad_dataset_to_batch_size(self, requests):
        dummy_instance = dataclasses.replace(requests[0], arguments=("hello", " there"), idx=len(requests))
        requests = requests + [dummy_instance] * (self.EvalBatch.size - len(requests) % self.EvalBatch.size)
        assert len(requests) % self.EvalBatch.size == 0
        dataset = EvalDataset(self.EvalPos, self.tokenizer, requests)
        return dataset

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

    Nones are not included in the dictionary representation, and LM Eval Harness will use its own defaults for any
    missing values.

    Docs are copied from the LM Eval Harness task guide. The LM Eval Harness task guide is the authoritative source
    for what these fields do. They were copied as of 2024-12-03.

    See Also:
       * [LM Eval Harness TaskConfig](https://github.com/EleutherAI/lm-evaluation-harness/blob/0ef7548d7c3f01108e7c12900a5e5eb4b4a668f7/lm_eval/api/task.py#L55)
       * [LM Eval Harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md#parameters)
    """

    task: str
    """ The name of the task to run."""
    task_alias: str | None = None
    """ An alias for the task. We log this name to wandb."""
    num_fewshot: int | None = None

    use_prompt: str | None = None
    """ Name of prompt in promptsource to use. if defined, will overwrite doc_to_text, doc_to_target, and doc_to_choice."""
    description: str | None = None
    """An optional prepended Jinja2 template or string which will be prepended to the few-shot examples passed into the model, often describing the task or providing instructions to a model, such as "The following are questions (with answers) about {{subject}}.\n\n". No delimiters or spacing are inserted between the description and the first few-shot example."""
    target_delimiter: str | None = None
    """String to insert between input and target output for the datapoint being tested. defaults to " " """
    fewshot_delimiter: str | None = None
    """ String to insert between few-shot examples. defaults to "\\n\\n" """
    doc_to_text: str | None = None
    """Jinja2 template string to process a sample into the appropriate input for the model."""
    doct_to_target: str | None = None
    """Jinja2 template string to process a sample into the appropriate target for the model."""
    doc_to_choice: str | None = None
    """Jinja2 template string to process a sample into a list of possible string choices for multiple_choice tasks. """

    def to_dict(self):
        base_dict = dataclasses.asdict(self)
        return {k: v for k, v in base_dict.items() if v is not None}


@dataclass(frozen=True)
class LmEvalHarnessConfig:
    task_spec: list[TaskConfig | str]
    max_examples: int | None = None
    max_eval_length: int | None = None
    log_samples: bool = False

    def to_task_spec(self) -> list[str | dict]:
        return [task.to_dict() if isinstance(task, TaskConfig) else task for task in self.task_spec]

    def to_task_dict(self) -> dict:
        """
        Convert the task spec to a dictionary that the LM Eval Harness expects.

        This is a bit more complex than we'd like, because we want to run e.g. Hellaswag 0-shot and 10-shot in the same
        run, and LM Eval Harness doesn't seem to want to do that by default. So we need to do some hacky stuff to make
        it work.
        """
        import lm_eval.tasks as tasks

        manager = tasks.TaskManager()
        # we need to do it this way b/c i can't figure out how to run e.g. hellaswag 0 shot and 10 shot in a single run
        this_tasks = {}
        for task in self.to_task_spec():
            try:
                if isinstance(task, str):
                    this_tasks.update(tasks.get_task_dict(task, manager))
                else:
                    our_name = task.get("task_alias", task["task"]) if isinstance(task, dict) else task
                    our_name = our_name.replace(" ", "_")
                    this_task = self._get_task_and_rename(manager, our_name, task)
                    this_tasks[our_name] = this_task
            except Exception:
                logger.exception(f"Failed to load task {task}")
                raise ValueError(f"Failed to load task {task}")
        return this_tasks

    def _get_task_and_rename(self, manager, our_name, task: dict | str):
        """
        Get a task from the task manager and rename it to our_name.
        LM Eval Harness doesn't seem to want to run multiple instances of the same task with different fewshot settings,
        (or other differences) so we need to hack around that.
        """
        import lm_eval.tasks as tasks

        task_dict = tasks.get_task_dict([task], manager)
        this_task = task_dict.popitem()[1]
        # hacky, but this allows us to run multiple instances of the same task with different fewshot settings
        this_task.config.task = our_name
        return this_task


@dataclass(frozen=True)
class EvalHarnessMainConfig:
    eval_harness: LmEvalHarnessConfig
    tokenizer: str
    checkpoint_path: str
    checkpoint_is_hf: bool = False
    """If True, the checkpoint is a HuggingFace checkpoint. Otherwise, it is a Levanter checkpoint."""
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)
    model: LmConfig = dataclasses.field(default_factory=Gpt2Config)

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


def _actually_run_eval_harness(
    config: LmEvalHarnessConfig,
    model: LmHeadModel,
    tasks_to_run: dict,
    tokenizer: HfTokenizer,
    EvalBatch: haliax.Axis,
    axis_resources: ResourceMapping,
):
    """
    Actually run the LM Eval Harness on the given model and tasks. This is a separate function so that it can be used
    by the main function and the callback function.

    Returns:
        The outputs of the LM Eval Harness with the following extra keys:
        - "averages": A dictionary with macro and micro averages for all metrics.

    """
    max_examples = config.max_examples
    max_eval_length = config.max_eval_length

    EvalPos = model.Pos if max_eval_length is None else model.Pos.resize(max_eval_length)
    harness = LevanterHarnessLM(EvalBatch, EvalPos, model, axis_resources, tokenizer)
    # we always set log_samples here and filter out the samples later if we don't want them
    outputs = evaluator.evaluate(harness, tasks_to_run, limit=max_examples, log_samples=True)

    averages = _compute_averages(outputs)
    outputs["averages"] = averages

    if not config.log_samples:
        del outputs["samples"]

    return outputs


def _compute_averages(outputs):
    """
    Compute macro and micro averages of all metrics.

    Args:
        outputs: Dictionary with results and samples:
                 - "results": Dictionary of task-level results.
                 - "samples": Dictionary of task-level sample counts.

    Returns:
        Averages dictionary with macro and micro averages for all metrics.
    """
    averages = {}
    metric_keys = set()

    # Collect all possible metrics across tasks
    for task_results in outputs["results"].values():
        metric_keys.update(k for k in task_results.keys() if "stderr" not in k and k != "alias")

    examples_per_task = [len(task_samples) for task_samples in outputs["samples"].values()]

    # Compute macro and micro averages
    for metric in metric_keys:
        # Collect valid tasks for this metric
        valid_tasks = [
            (task_results.get(metric), examples_per_task[i])
            for i, (task_name, task_results) in enumerate(outputs["results"].items())
            if metric in task_results
        ]

        if not valid_tasks:
            continue  # Skip metrics with no valid tasks

        # Separate metric values and weights
        metric_values, this_examples_per_task = zip(*valid_tasks)

        # Compute macro and micro averages
        averages["macro_avg_" + metric] = np.mean(metric_values)
        averages["micro_avg_" + metric] = np.average(metric_values, weights=this_examples_per_task)

    return averages


BITS_PER_NAT = 1 / np.log(2)

# eval_harness isn't consistent enough for this to actually be workable
# def _compute_extra_metrics(samples):
#     """
#     Compute a few "soft" measures of accuracy for each task, based on the outputs of the eval harness.
#
#     Specifically, we compute:
#        - "bpb": bits per byte of the correct completion
#        - "logprob": log probability of the correct completion
#        - "choice_logprob": log probability of the correct choice normalized w.r.t. the other choices
#        - "choice_prob_norm": probability of the length-normalized correct choice normalized w.r.t. the other choices
#
#     Args:
#         samples: Dictionary with task data, where each task has a list of samples. Each sample contains:
#                  - "doc": The original task document (can include metadata such as the answer key)
#                  - "target": Index of the correct answer (0-indexed), or
#                              "doc.answer" for 1-indexed answers.
#                  - "arguments": List of [input, completion] pairs
#                  - "resps": List of [log probability, is_correct] pairs for completions
#
#     Returns:
#         A dictionary with per-task aggregated metrics.
#     """
#     # TODO: move to eval harness and use more sane logic
#     # uses the samples which has one of two structures (that I've seen)
#     # { "<task>": [ {"doc": {...,}, "target": <0-indexed answer>, "arguments": [[input, completion], "resps": [[score, is_correct], ...], ...}, ...] }
#     # { "<task>": [ {"doc": {..., "answer": "[1-indexed answer]"}, "target": "<useless string>", "arguments": [input, completion], "resps": [[score, is_correct], ...], ...}, ...] }
#     metrics = {}
#
#     for task, samples in samples.items():
#         bpb_list = []
#         logprob_list = []
#         choice_logprob_list = []
#         choice_prob_norm_list = []
#
#         for sample in samples:
#             # Extract the correct answer index (supporting both 0-indexed `target` and 1-indexed `doc.answer`)
#             if "answer" in sample["doc"]:
#                 target = int(sample["doc"]["answer"]) - 1  # Convert 1-indexed to 0-indexed
#             elif "label" in sample["doc"]:
#                 target = int(sample["doc"]["label"])
#             elif "target" in sample and isinstance(sample["target"], int):
#                 target = sample["target"]  # 0-indexed target
#             elif "target" in sample and isinstance(sample["target"], str):
#                 # see if it's A-Z:
#                 if len(sample["target"]) == 1 and "A" <= sample["target"] <= "Z":
#                     target = ord(sample["target"]) - ord("A")
#                 else:
#                     raise ValueError(f"Invalid target: {sample['target']}. {sample}")
#             elif "target" in sample and isinstance(sample["target"], list):
#                 target = sample["target"][0]
#             else:
#                 raise KeyError(f"Missing `target` or `doc.answer` in sample. doc id: {sample['doc_id']}. Hash: {sample['doc_hash']}\n\n{sample}")
#
#             resps = sample["filtered_resps"]  # List of [log probability, is_correct]
#             arguments = sample["arguments"]  # [input, completion] pairs
#
#             # Compute byte lengths for each choice
#             byte_lengths = [max(1, len(completion.encode("utf-8"))) for _, completion in arguments]
#
#             # Compute log probabilities for each choice
#             log_probs = np.array([resp[0] for resp in resps])  # Extract log probabilities
#             assert log_probs.shape == (len(arguments),), f"Log probs shape: {log_probs.shape}, arguments: {len(arguments)}. doc: {sample}"
#             normalized_log_probs = log_probs - np.logaddexp.reduce(log_probs)
#
#             # Metrics for the correct answer
#             correct_logprob = log_probs[target]
#             correct_bpb = -correct_logprob / byte_lengths[target] * NAT_TO_BIT
#             correct_choice_logprob = normalized_log_probs[target]
#
#             # Compute length-normalized weights (w_i)
#             bpb_values = -log_probs / np.array(byte_lengths) * NAT_TO_BIT
#             bpb_weights = np.exp(-bpb_values)
#             bpb_weights /= max(bpb_weights.sum(), 1e-8)  # Avoid division by zero
#             correct_choice_prob_norm = bpb_weights[target]
#
#             # Append metrics
#             bpb_list.append(correct_bpb)
#             logprob_list.append(correct_logprob)
#             choice_logprob_list.append(correct_choice_logprob)
#             choice_prob_norm_list.append(correct_choice_prob_norm)
#
#         # Aggregate metrics for the task
#         metrics[task] = {
#             "bpb": np.mean(bpb_list) if bpb_list else 0.0,
#             "logprob": np.mean(logprob_list) if logprob_list else 0.0,
#             "choice_logprob": np.mean(choice_logprob_list) if choice_logprob_list else 0.0,
#             "choice_prob_norm": np.mean(choice_prob_norm_list)  if choice_prob_norm_list else 0.0,
#         }
#
#     return metrics


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
        with open("lm_eval_harness_results.json", "w") as f:
            json.dump(outputs, f, indent=2)

        # also write to stdout
        if jax.process_index() == 0:
            print(json.dumps(outputs, indent=2), flush=True)

        # also log the results
        levanter.tracker.current_tracker().log_artifact("lm_eval_harness_results.json", name="lm_eval_harness_results")
        log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())

        return outputs


def log_report_to_tracker(prefix: str, report: dict, tracker: Optional[levanter.tracker.Tracker] = None):
    if tracker is None:
        tracker = levanter.tracker.current_tracker()

    to_log = {}
    for task_name, task_results in report["results"].items():
        for metric_name, metric_value in task_results.items():
            # remove the ",none" suffix, which eval-harness adds by default for some reason
            if metric_name.endswith(",none"):
                metric_name = metric_name[:-5]

            if isinstance(metric_value, float | int):
                to_log[f"{prefix}/{task_name}/{metric_name}"] = metric_value

    if "averages" in report:
        for metric_name, metric_value in report["averages"].items():
            if isinstance(metric_value, float | int):
                if metric_name.endswith(",none"):
                    metric_name = metric_name[:-5]

                to_log[f"{prefix}/averages/{metric_name}"] = metric_value

    tracker.log(to_log, step=None)


def lm_eval_harness(config: LmEvalHarnessConfig, tokenizer, EvalBatch, axis_resources):
    tasks_to_run = config.to_task_dict()

    def lm_eval_harness(step: StepInfo, force=False):
        if step.step == 0 and not force:
            return  # don't run eval on the first step

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
            # don't delete b/c wandb will sometimes defer upload
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
                import json

                json.dump(outputs, f)
                levanter.tracker.current_tracker().log_artifact(
                    f.name, name=f"lm_eval_harness_results.{step.step}.json", type="lm_eval_output"
                )

            log_report_to_tracker("lm_eval", outputs, levanter.tracker.current_tracker())

    return lm_eval_harness


if __name__ == "__main__":
    levanter.config.main(run_eval_harness_main)()
    print("Done", flush=True)
