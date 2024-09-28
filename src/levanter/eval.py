import dataclasses
import logging
import warnings
from typing import Callable, Optional, Sequence, TypeVar

import jax.numpy as jnp
import jmp
import numpy as np
import tqdm
from jax.sharding import Mesh

import haliax as hax
from haliax.partitioning import ResourceMapping

import levanter.tracker
from levanter.data import Dataset, ReplicatedBatchLoader
from levanter.logging import LoadingTimeTrackerIterator
from levanter.models.lm_model import LmExample, LmHeadModel, compute_next_token_loss
from levanter.trainer import StepInfo
from levanter.utils.stat_utils import RunningMean
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


T = TypeVar("T")
M = TypeVar("M")


@dataclasses.dataclass
class EvalResult:
    micro_avg_loss: float  # per token across all datasets
    macro_avg_loss: float  # average of per-dataset average losses
    tag_macro_losses: dict[str, float]  # per tag average-per-token loss
    tag_micro_losses: dict[str, float]  # per tag total loss, for "parent" tags
    total_eval_loading_time: float


class DomainTaggedDataset(Dataset[tuple[T, hax.NamedArray]]):
    """Holds multiple datasets, each with its own domain tag. Also indexes the tags to enable easier aggregation."""

    @property
    def tags(self):
        return self.tag_to_index.keys()

    def __init__(
        self, datasets: Sequence[tuple[Dataset[T], Sequence[str]]], max_examples_per_dataset: Optional[int] = None
    ):
        self.datasets = []
        tag_index: dict[str, int] = {}
        for i, (dataset, tags) in enumerate(datasets):
            if tags is None:
                warnings.warn("Dataset has no tags. Giving it an index")
                tags = [f"domain_{i}"]
            for tag in tags:
                if tag not in tag_index:
                    tag_index[tag] = len(tag_index)

            self.datasets.append((dataset, tags))

        self.tag_to_index = tag_index
        self.Tag = hax.Axis("tag", len(self.tag_to_index))
        self.max_examples_per_dataset = max_examples_per_dataset

    def __iter__(self):
        for dataset, tags in self.datasets:
            indexed = [self.tag_to_index[tag] for tag in tags]
            tags = np.zeros(self.Tag.size, dtype=np.int32)
            tags[indexed] = 1
            tags = hax.named(tags, self.Tag)

            count = 0
            for example in dataset:
                if self.max_examples_per_dataset is not None and count >= self.max_examples_per_dataset:
                    break
                count += 1
                yield example, tags


def _join_prefix(prefix: str, tag: str) -> str:
    if prefix:
        return f"{prefix}/{tag}"
    return tag


def cb_tagged_lm_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets: Sequence[tuple[Dataset[LmExample], Sequence[str]]],
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping = None,
    max_examples_per_dataset: Optional[int] = None,
    prefix: str = "eval",
    mp: jmp.Policy = None,
) -> Callable[[StepInfo], EvalResult]:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting
    an overall score.

    Tags can be hierarchical, with "/" as a separator. We log both a micro and macro average loss
    for each tag.

    !!! note

        loss_fn should return *per-token* loss (shape [EvalBatch, Token])

    Args:
        EvalBatch: The axis for the evaluation batch (mostly for the batch size)
        tagged_eval_sets: A list of datasets, each with its own domain tag
        device_mesh: The mesh to use for evaluation
        axis_mapping: The axis mapping to use for evaluation
    """

    evaluator = TaggedEvaluator(
        EvalBatch, tagged_eval_sets, device_mesh, axis_mapping, max_examples_per_dataset, mp=mp
    )

    def eval_callback(step: StepInfo):
        with levanter.tracker.capture_time() as time_fn:
            result = evaluator.evaluate(step.model)

        log_dict = {
            # log micro average as just "loss"
            _join_prefix(prefix, "loss"): result.micro_avg_loss,
            _join_prefix(prefix, "macro_loss"): result.macro_avg_loss,
            _join_prefix(prefix, "loading_time"): result.total_eval_loading_time,
            _join_prefix(prefix, "total_time"): time_fn(),
        }

        logger.info(f"{prefix} loss: {result.micro_avg_loss:.3f}")
        for tag, loss in result.tag_macro_losses.items():
            # don't log leaf tag macro losses because it doesn't mean anything different than micro loss
            if tag in evaluator.dataset.tag_to_index:
                continue
            if not tag:
                continue
            log_dict[_join_prefix(prefix, tag) + "/macro_loss"] = loss
            logger.info(f"{tag} macro loss: {loss:.3f}")

        for tag, loss in result.tag_micro_losses.items():
            if not tag:
                continue
            if tag in evaluator.dataset.tag_to_index:
                log_dict[_join_prefix(prefix, tag) + "/loss"] = loss
                logger.info(f"{tag} loss: {loss:.3f}")
            else:
                log_dict[_join_prefix(prefix, tag) + "/micro_loss"] = loss
                logger.info(f"{tag} micro loss: {loss:.3f}")

        levanter.tracker.log_metrics(log_dict, step=step.step)

        return result

    return eval_callback


class TaggedEvaluator:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting an overall score.

    Tags are arranged hierarchically with "/" as separator, and we log both a micro and macro average loss
    for each tag.

    """

    def __init__(
        self,
        EvalBatch: hax.Axis,
        tagged_eval_sets,
        device_mesh=None,
        axis_mapping=None,
        max_examples_per_dataset=None,
        mp: Optional[jmp.Policy] = None,
    ):
        self.EvalBatch = EvalBatch
        self.dataset = DomainTaggedDataset(tagged_eval_sets, max_examples_per_dataset)
        self.loader = ReplicatedBatchLoader(
            self.dataset, mesh=device_mesh, axis_resources=axis_mapping, Batch=EvalBatch
        )
        self.mp = mp

        # tags are arranged hierarchically with "/" as separator. We want to log the average loss for each tag.
        hierarchy: dict[str, list[int]] = {}
        for tag, index in self.dataset.tag_to_index.items():
            parts = tag.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                assert parent != tag
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(index)

        self.hierarchy = hierarchy

        @hax.named_jit(out_axis_resources=axis_mapping)
        def accum_for_batch(
            m: LmHeadModel, state: tuple[RunningMean, RunningMean], batch: LmExample, tags: hax.NamedArray
        ):
            m = inference_mode(m, True)

            if self.mp is not None:
                m = self.mp.cast_to_compute(m)
            with hax.axis_mapping(axis_mapping):
                total_mean, mean_per_tag = state
                losses = compute_next_token_loss(m, batch, reduction=None, reduction_axis=())
                mask = batch.loss_mask  # [Batch, Token]
                this_tokens = hax.sum(mask)
                this_loss = hax.einsum("->", losses, mask)  # to scalar

                this_tokens_per_tag = hax.einsum("-> tag", mask, tags)
                this_loss_per_tag = hax.einsum("-> tag", mask, losses, tags)  # [Tag]

                mean = total_mean.add(this_loss / this_tokens, this_tokens)
                # careful: this_tokens_per_tag can be 0 if there are no tokens for that tag
                safe_mean = hax.where(this_tokens_per_tag, this_loss_per_tag / this_tokens_per_tag, 0.0)
                mean_per_tag = mean_per_tag.add(safe_mean, this_tokens_per_tag)

            return mean, mean_per_tag

        self.accum_for_batch = accum_for_batch

    def evaluate(self, m: LmHeadModel):
        total_loss = jnp.zeros(())
        mean_losses_per_tag = hax.zeros(self.dataset.Tag, dtype=np.float32)

        state = (RunningMean.zeros_like(total_loss), RunningMean.zeros_like(mean_losses_per_tag))
        state = hax.shard(state)

        iterator = LoadingTimeTrackerIterator(self.loader)

        for batch, tags in tqdm.tqdm(iterator, "eval"):
            state = self.accum_for_batch(m, state, batch, tags)

        total_loss, losses_per_tag = state

        micro_avg_loss = total_loss.mean.item()
        tag_avg_loss = losses_per_tag.mean

        # TODO: why do i have to jit this
        macro_avg_loss = hax.named_jit(lambda x: hax.mean(x).array)(tag_avg_loss).item()

        tag_macro_loss = {}
        tag_micro_loss = {}

        mean_loss_per_tag_cpu = np.array(losses_per_tag.mean.array)  # type: ignore
        total_tokens_per_tag_cpu = np.array(losses_per_tag.total.array)  # type: ignore

        # add in the hierarchy
        for parent, children in self.hierarchy.items():
            mask = np.zeros(self.dataset.Tag.size, dtype=bool)
            mask[children] = 1
            assert total_tokens_per_tag_cpu.shape == mask.shape

            # don't consider tags with no tokens in macro average
            mask = mask & (total_tokens_per_tag_cpu > 0)

            # macro is the average of the averages
            tag_macro_loss[parent] = np.mean(mean_loss_per_tag_cpu, where=mask)
            # micro is the total loss for the parent tag
            # (average doesn't support where directly so we just 0 out the weights)
            tag_micro_loss[parent] = np.average(mean_loss_per_tag_cpu, weights=total_tokens_per_tag_cpu * mask)

        for tag, index in self.dataset.tag_to_index.items():
            tag_micro_loss[tag] = mean_loss_per_tag_cpu[index]
            # no macro loss for the leaf tags

        return EvalResult(micro_avg_loss, macro_avg_loss, tag_macro_loss, tag_micro_loss, iterator.total_time)
