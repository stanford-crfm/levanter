import dataclasses
from typing import Callable, Iterator, Optional, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

import levanter.tracker
from haliax import Scalar, named_jit
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax.types import IntScalar

from levanter.data import ShardableDataset
from levanter.data.mixture import MixtureDataset
from levanter.trainer import M, StepInfo, Trainer, TrainerConfig, TrainerState
from levanter.types import ComputeLossFunction
from optax._src.base import GradientTransformation

from levanter.utils.tree_utils import inference_mode

T = TypeVar("T")

class DoremiState(TrainerState):
    alpha: hax.NamedArray
    average_alpha: hax.NamedArray

    def update_alpha(self, alpha):
        average_alpha = self.average_alpha + (alpha - self.average_alpha) / (self.step + 1)
        return dataclasses.replace(self, alpha=alpha, average_alpha=average_alpha)


class DoReMiTrainer(Trainer):
    # we just use the DoReMi trainer for state management

    def _initialize_state_from_scratch(self, model_init: Callable[[], M], training_key, is_trainable):
        base_state = super()._initialize_state_from_scratch(model_init, training_key, is_trainable)
        return DoremiState(base_state.step,
                           base_state.model,
                           base_state.opt_state,
                           base_state.training_key,
                           self.initial_alpha)


def estimate_mixture_weights(
    trainer: TrainerConfig,
    loss_fn: ComputeLossFunction,
    initial_proxy,
    ref,
    data_sources: dict[str, ShardableDataset[T]],
    ref_weights: dict[str, float],
    *,
    domain_weight_step: float = 1.0,
    smoothing: float = 1e-3,
    eps_alpha: float = 1e-6,
    key: PRNGKeyArray,
) -> dict[str, float]:
    """
    Estimate the mixture weights for the data sources using DoReMi.
    https://arxiv.org/abs/2305.10429
    """
    training_key, data_key = jrandom.split(key)
    domain_indices = list(data_sources.keys())
    domain_to_index = {domain: index for index, domain in enumerate(domain_indices)}

    # Initialize domain weights.
    # TODO: should we initialize to the ref or to uniform?
    Domain = hax.Axis("domain", len(domain_indices))
    initial_alpha = hax.ones(Domain) / Domain.size

    trainer = DoReMiTrainer(trainer, optax.adamw(1e-3), ref, initial_alpha, loss_fn)

    # calculate per-token losses for proxy and ref
    def compute_excess_loss(ref, proxy, batch):
        proxy_losses = loss_fn(proxy, batch, reduction_axis=())
        ref_losses = loss_fn(ref, batch, reduction_axis=())
        # calculate excess losses
        excess_losses = proxy_losses - ref_losses
        return excess_losses


    # Loss is alpha_d * (proxy - ref) (basically the unclipped excess loss with the new alpha)
    def proxy_model_loss(excess_losses, domains, alpha):
        one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
        # basically einsum(" * -> ", alpha, one_hot_domains, excess_losses)
        # TODO: I'd like to make the syntax for this nicer. einsum would be like
        # einsum("d,bd,b... -> ()" ro something)
        # but it's really just collapsing all axes
        loss = hax.dot(excess_losses.axes + (Domain,), alpha, one_hot_domains, excess_losses).scalar()

        return loss

    @hax.named_jit(axis_resources=trainer.parameter_axis_mapping, donate_args=(True, ))
    def doremi_step(state: DoremiState, batch, domains):
        proxy = inference_mode(state.model, False)
        # this is one of those times when PyTorch's backward() is nice
        excess_losses, excess_backward = eqx.filter_vjp(lambda proxy: compute_excess_loss(proxy, ref, batch), proxy)

        # Update domain weights
        ## Compute per-domain excess losses
        clipped_losses = hax.maximum(excess_losses, 0)
        one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
        per_domain_losses = hax.dot(excess_losses.axes, one_hot_domains, clipped_losses)

        alpha = state.alpha * hax.exp(domain_weight_step * per_domain_losses)
        alpha /= hax.sum(alpha)
        alpha = (1 - smoothing) * alpha + initial_alpha * smoothing

        alpha_distance = hax.sum(hax.abs(alpha - state.alpha))
        levanter.tracker.log_metrics({"alpha_distance": alpha_distance}, step=state.step)

        # Update proxy model weights θt for the objective L(θt−1, αt) (using Adam, Adafactor, etc.)
        loss, grad_loss = eqx.filter_value_and_grad(proxy_model_loss)(excess_losses, domains, alpha)
        grad = excess_backward(grad_loss)

        new_state = trainer._take_train_step(state, proxy, grad, batch)

        return loss, new_state

    # TODO: we don't support serializing stuff from anything other than the model and the opt_state. should fix.
    running_alpha_mean = initial_alpha

    # we're not actually going to use the trainer for very much but it holds hooks and sets up contexts
    with trainer:
        tagged_mixture = domain_tagged_mixture(data_sources, ref_weights, domain_to_index, key=data_key)
        state = trainer.initial_state(training_key, model=initial_proxy)
        del initial_proxy
        train_loader = iter(trainer.sharded_loader(tagged_mixture, trainer.TrainBatch))

        if state.step > 0:
            # step is after the batch, so we need to seek to step
            # TODO: implement iter_data.seek(resume_step +1)
            import tqdm

            for _ in tqdm.tqdm(range(state.step + 1), desc="seeking data for resume"):
                next(train_loader)

        while state.step < trainer.num_train_steps:
            example, ex_domains = next(train_loader)

            key, new_key = jax.random.split(state.training_key)
            proxy, alpha = state.model

            loss, new_model, new_optstate = doremi_step(
                proxy, state.opt_state, alpha, example, ex_domains,
            )
            loss = loss.item()  # type: ignore

            new_info = StepInfo(TrainerState(state.step + 1, new_model, new_optstate, new_key), loss, step_time())

            trainer.run_hooks(new_info)

            state = new_info








def domain_tagged_mixture(
    data_sources: dict[str, ShardableDataset[T]],
    weights: dict[str, float],
    domain_to_index: dict[str, int],
    *,
    key: PRNGKeyArray,
) -> MixtureDataset[(T, IntScalar)]:
    """
    Domain tagged mixture dataset. This dataset will yield from the datasets according to the weights,
    and will yield the domain index as a second element of the tuple.
    """
    tagged_datasets = {
        domain_index: DomainTaggedDataset(data_sources[domain], domain_index)
        for domain, domain_index in domain_to_index.items()
    }

    return MixtureDataset(tagged_datasets, weights, key=key)


class DomainTaggedDataset(ShardableDataset[(T, hax.NamedArray)]):  # named array is a scalar int
    def __init__(
        self,
        dataset: ShardableDataset[T],
        domain_index: int | hax.NamedArray,
    ):
        self.dataset = dataset

        if isinstance(domain_index, int):
            self.domain_index = hax.named(jnp.array(domain_index, dtype=int), ())
        else:
            self.domain_index = domain_index

    def shard(self, shard_id: int, num_shards: int) -> "DomainTaggedDataset[T]":
        return DomainTaggedDataset(self.dataset.shard(shard_id, num_shards), self.domain_index)

    def __iter__(self) -> Iterator[(T, IntScalar)]:
        for item in self.dataset:
            yield item, self.domain_index
