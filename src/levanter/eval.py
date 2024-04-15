import dataclasses
from typing import Callable, Optional, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import haliax as hax
from haliax.partitioning import ResourceMapping

import levanter.tracker
from levanter.data import Dataset, ReplicatedBatchLoader
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.trainer import StepInfo
from levanter.utils.tree_utils import inference_mode


T = TypeVar("T")
M = TypeVar("M")


@dataclasses.dataclass
class EvalResult:
    micro_avg_loss: float  # per token across all datasets
    macro_avg_loss: float  # average of per-dataset average losses
    tag_macro_losses: dict[str, float]  # per tag average-per-token loss
    tag_micro_losses: dict[str, float]  # per tag total loss, for "parent" tags


class DomainTaggedDataset(Dataset[tuple[T, hax.NamedArray]]):
    """Holds multiple datasets, each with its own domain tag. Also indexes the tags to enable easier aggregation."""

    @property
    def tags(self):
        return self.tag_to_index.keys()

    def __init__(self, datasets: Sequence[tuple[Dataset[T], Sequence[str]]]):
        self.datasets = datasets
        tag_index = {"__ALL__": 0}
        for _, tags in datasets:
            for tag in tags:
                if tag not in tag_index:
                    tag_index[tag] = len(tag_index)

        self.tag_to_index = tag_index
        self.Tag = hax.Axis("tag", len(self.tag_to_index))

    def __iter__(self):
        for dataset, tags in self.datasets:
            indexed = [self.tag_to_index[tag] for tag in tags] + [0]
            # tags = indexed + [0] * (self.max_tags_per_dataset - len(indexed))
            # tags = np.array(tags, dtype=np.int32)
            tags = np.zeros(len(self.tag_to_index), dtype=np.int32)
            tags[
                tuple(
                    indexed,
                )
            ] = 1
            tags = hax.named(tags, self.Tag)

            for example in dataset:
                yield example, tags


def cb_tagged_lm_evaluate(
    EvalBatch: hax.Axis,
    tagged_eval_sets: Sequence[tuple[Dataset[LmExample], Sequence[str]]],
    device_mesh: Optional[Mesh] = None,
    axis_mapping: ResourceMapping = None,
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

    evaluator = TaggedEvaluator(EvalBatch, tagged_eval_sets, device_mesh, axis_mapping)

    def eval_callback(step: StepInfo):
        result = evaluator.evaluate(step.model)
        levanter.tracker.log_metrics({"eval/micro_avg_loss": result.micro_avg_loss}, step=step.step)
        levanter.tracker.log_metrics({"eval/macro_avg_loss": result.macro_avg_loss}, step=step.step)
        for tag, loss in result.tag_macro_losses.items():
            levanter.tracker.log_metrics({f"eval/{tag}/macro_loss": loss}, step=step.step)

        for tag, loss in result.tag_micro_losses.items():
            levanter.tracker.log_metrics({f"eval/{tag}/micro_loss": loss}, step=step.step)

        return result

    return eval_callback


class TaggedEvaluator:
    """
    Evaluates multiple tagged datasets using a given evaluation function.
    Scores for each tag are aggregated and logged separately, as well as getting an overall score.

    Tags are arranged hierarchically with "/" as separator, and we log both a micro and macro average loss
    for each tag.

    """

    def __init__(self, EvalBatch: hax.Axis, tagged_eval_sets, device_mesh=None, axis_mapping=None):
        self.EvalBatch = EvalBatch
        self.dataset = DomainTaggedDataset(tagged_eval_sets)
        self.loader = ReplicatedBatchLoader(
            self.dataset, mesh=device_mesh, axis_resources=axis_mapping, Batch=EvalBatch
        )

        # tags are arranged hierarchically with "/" as separator. We want to log the average loss for each tag.
        hierarchy: dict[str, list[int]] = {}
        for tag, index in self.dataset.tag_to_index.items():
            parts = tag.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                assert parent != tag
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(i)

        self.hierarchy = hierarchy

        @hax.named_jit
        def accum_for_batch(m: LmHeadModel, state, batch: LmExample, tags: hax.NamedArray):
            m = inference_mode(m, True)
            with hax.axis_mapping(axis_mapping):
                total_loss, total_tokens, losses_per_tag, total_tokens_per_tag = state
                losses = m.compute_loss(batch, reduction=None, reduction_axis=())
                mask = batch.loss_mask  # [Batch, Token]
                # TODO: running mean?
                total_loss += hax.einsum("->", losses, mask)  # to scalar
                total_tokens += hax.einsum("->", mask)
                losses_per_tag += hax.einsum("-> tag", losses, mask, tags)
                total_tokens_per_tag += hax.einsum("-> tag", mask, tags)

            return total_loss, total_tokens, losses_per_tag, total_tokens_per_tag

        self.accum_for_batch = accum_for_batch

    def evaluate(self, m: LmHeadModel):
        total_loss = jnp.zeros(())
        total_tokens = 0
        losses_per_tag = hax.zeros(self.dataset.Tag, dtype=np.float32)
        total_tokens_per_tag = hax.zeros(self.dataset.Tag, dtype=np.int32)

        state = (total_loss, total_tokens, losses_per_tag, total_tokens_per_tag)
        for batch, tags in self.loader:
            state = self.accum_for_batch(m, state, batch, tags)

        total_loss, total_tokens, losses_per_tag, total_tokens_per_tag = state

        micro_avg_loss = (total_loss / total_tokens).item()
        tag_avg_loss = losses_per_tag / total_tokens_per_tag
        macro_avg_loss = hax.mean(tag_avg_loss).item()

        tag_macro_loss = {}
        tag_micro_loss = {}

        losses_per_tag_cpu = np.array(losses_per_tag.array)
        total_tokens_per_tag_cpu = np.array(total_tokens_per_tag.array)

        # add in the hierarchy
        for parent, children in self.hierarchy.items():
            children = np.array(children)
            tag_macro_loss[parent] = np.sum(losses_per_tag_cpu, where=children) / np.sum(
                total_tokens_per_tag_cpu, where=children
            )
            tag_micro_loss[parent] = np.mean(losses_per_tag_cpu, where=children)

        for tag, index in self.dataset.tag_to_index.items():
            loss = losses_per_tag_cpu[index] / total_tokens_per_tag_cpu[index]
            tag_macro_loss[tag] = loss
            # no micro loss for the parent tags

        return EvalResult(micro_avg_loss, macro_avg_loss, tag_macro_loss, tag_micro_loss)
