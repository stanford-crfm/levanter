import dataclasses
import logging
from typing import Callable, Iterator, Optional, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax.types import IntScalar

import levanter.tracker
from levanter.data import ShardableDataset
from levanter.data.mixture import MixtureDataset
from levanter.logging import capture_time
from levanter.trainer import M, StepInfo, Trainer, TrainerConfig, TrainerState
from levanter.types import ComputeLossFunction, ModuleComputeLoss
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


T = TypeVar("T")


# TODO: should we put ref in the state? If so, need to tell it to not serialize it
class DoremiState(TrainerState):
    alpha: hax.NamedArray
    average_alpha: hax.NamedArray

    def update_alpha(self, alpha):
        average_alpha = self.average_alpha + (alpha - self.average_alpha) / (self._step + 1)
        return dataclasses.replace(self, alpha=alpha, average_alpha=average_alpha)


class DoReMiTrainer(Trainer):
    # we just use the DoReMi trainer for state management

    def __init__(self, trainer: TrainerConfig, optimizer: optax.GradientTransformation, initial_alpha: hax.NamedArray):
        super().__init__(trainer, optimizer)
        self.initial_alpha = initial_alpha

    # TODO: I'd like to not need to override trainer for this
    def _initialize_state_from_scratch(self, model: Callable[[], M], training_key, is_trainable):
        base_state = super()._initialize_state_from_scratch(model, training_key, is_trainable)

        return DoremiState(**base_state.__dict__, alpha=self.initial_alpha, average_alpha=self.initial_alpha)


@dataclasses.dataclass
class DoReMiConfig:
    # This is designed to be used with estimate_mixture_weights
    domain_weight_step_size: float = 1.0
    smoothing: float = 1e-3
    sampling_weights: Optional[dict[str, float]] = None


DEFAULT_DOREMI_TRAINER_CONFIG = TrainerConfig(
    num_train_steps=10000,
    train_batch_size=512,
)


def estimate_mixture_weights(
    initial_proxy: M,
    ref: M,
    data_sources: dict[str, ShardableDataset[T]],
    sampling_weights: Optional[dict[str, float]] = None,
    *,
    trainer_config: TrainerConfig = DEFAULT_DOREMI_TRAINER_CONFIG,
    loss_fn: ComputeLossFunction[M, T] = ModuleComputeLoss(),
    domain_weight_step_size: float = 1.0,
    smoothing: float = 1e-3,
    weight_change_eps: float = 1e-3,
    key: PRNGKeyArray,
) -> dict[str, float]:
    """
    Estimate the mixture weights for the data sources using DoReMi.
    https://arxiv.org/abs/2305.10429

    Args:
        trainer_config: Trainer config
        initial_proxy: Initial proxy model
        ref: Reference model
        data_sources: Data sources to estimate the weights for
        sampling_weights: Sampling weights for the data sources. If not provided, will use uniform sampling weights.
        loss_fn: Loss function to use for the proxy and ref models. If not provided, will use the model's compute_loss
        domain_weight_step_size: Step size for the domain weights
        smoothing: Smoothing for the domain weights
        key: PRNG key
    """
    if len(data_sources) <= 1:
        raise ValueError("Must have at least two data sources")

    training_key, data_key = jrandom.split(key)
    domain_indices = list(data_sources.keys())
    domain_to_index = {domain: index for index, domain in enumerate(domain_indices)}

    # Initialize domain weights.
    # TODO: should we initialize to the ref or to uniform?
    Domain = hax.Axis("domain", len(domain_indices))
    initial_alpha = hax.ones(Domain) / Domain.size

    trainer = DoReMiTrainer(trainer_config, optax.adamw(1e-3), initial_alpha)
    with trainer:
        ref = _prepare_ref_model(ref, trainer_config)

    if sampling_weights is not None:
        assert set(sampling_weights.keys()) == set(data_sources.keys())
        sampling_weights = {
            domain: weight / sum(sampling_weights.values()) for domain, weight in sampling_weights.items()
        }
    else:
        sampling_weights = {domain: 1 / len(data_sources) for domain in data_sources.keys()}

    # calculate per-token losses for proxy and ref
    def compute_excess_loss(proxy, ref, batch):
        proxy_losses = loss_fn(proxy, batch, reduction_axis=())
        ref_losses = loss_fn(ref, batch, reduction_axis=())
        # calculate excess losses
        excess_losses = proxy_losses - ref_losses
        return excess_losses

    # Loss is \sum_d alpha_d * (proxy - ref) (basically the unclipped excess loss with the new alpha)
    @hax.named_jit(axis_resources=trainer.parameter_axis_mapping, donate_args=(True,))
    def doremi_step(state: DoremiState, ref, batch, domains):
        proxy = inference_mode(state.model, False)
        with hax.axis_mapping(trainer.compute_axis_mapping):
            # this is one of those times when PyTorch's backward() is nice
            excess_losses, excess_backward = eqx.filter_vjp(
                lambda proxy: compute_excess_loss(proxy, ref, batch), proxy
            )

            clipped_losses = hax.maximum(excess_losses, 0)

            per_domain_losses = _compute_per_domain_losses(Domain, domains, clipped_losses)

            # Update domain weights
            alpha = state.alpha * hax.exp(domain_weight_step_size * per_domain_losses)
            alpha /= hax.sum(alpha)
            alpha = (1 - smoothing) * alpha + initial_alpha * smoothing

            distance_from_uniform = hax.sum(hax.abs(alpha - initial_alpha))

            # Update proxy model weights θt for the objective L(θt−1, αt) (using Adam, Adafactor, etc.)
            loss, grad_loss = eqx.filter_value_and_grad(_domain_weighted_loss)(excess_losses, Domain, domains, alpha)
            grad = excess_backward(grad_loss)[0]

        new_state = trainer._take_train_step(state, proxy, grad)
        new_state = new_state.update_alpha(alpha)

        alpha_distance = hax.sum(hax.abs(new_state.average_alpha - state.average_alpha))
        alpha_dict = _alpha_weights_to_dict(Domain, new_state.average_alpha, domain_to_index)

        levanter.tracker.jit_log_metrics(
            {
                "change_in_alpha": alpha_distance,
                "alpha_distance_from_uniform": distance_from_uniform,
                "alpha": alpha_dict,
            },
            step=state._step,
        )

        return loss, alpha_distance, new_state

    # we're not actually going to use the trainer for very much but it holds hooks and sets up contexts
    with trainer:
        tagged_mixture = domain_tagged_mixture(data_sources, sampling_weights, domain_to_index, key=data_key)
        state: DoremiState = trainer.initial_state(training_key, model=initial_proxy)
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

            with capture_time() as step_time:
                loss, alpha_distance, state = doremi_step(state, ref, example, ex_domains)
                loss = loss.item()  # type: ignore

            new_info = StepInfo(state, loss, step_time())

            trainer.run_hooks(new_info)

            # check convergence for alphas
            if alpha_distance.item() < weight_change_eps:
                logger.info(f"Converged on alpha at step {state.step}: {alpha_distance:.4f}")
                break

        trainer.run_hooks(new_info, force=True)

        alpha = state.average_alpha
        final_weights = _alpha_weights_to_dict(Domain, alpha, domain_to_index)

        levanter.tracker.log_summary({"final_alpha": final_weights})

    return {k: float(v) for k, v in final_weights.items()}


def _alpha_weights_to_dict(Domain, alpha, domain_name_to_index):
    final_weights = {domain: alpha[Domain, index] for domain, index in domain_name_to_index.items()}
    return final_weights


def _prepare_ref_model(ref, trainer):
    return hax.named_jit(
        lambda m: trainer.mp.cast_to_compute(inference_mode(m, True)),
        axis_resources=trainer.parameter_axis_mapping,
        donate_args=True,
    )(ref)


def domain_tagged_mixture(
    data_sources: dict[str, ShardableDataset[T]],
    weights: dict[str, float],
    domain_to_index: dict[str, int],
    *,
    key: PRNGKeyArray,
) -> MixtureDataset[Tuple[T, IntScalar]]:
    """
    Domain tagged mixture dataset. This dataset will yield from the datasets according to the weights,
    and will yield the domain index as a second element of the tuple.
    """
    tagged_datasets = {
        domain: DomainTaggedDataset(data_sources[domain], domain_index)
        for domain, domain_index in domain_to_index.items()
    }

    return MixtureDataset(tagged_datasets, weights, key=key)


class DomainTaggedDataset(ShardableDataset[Tuple[T, hax.NamedArray]]):  # named array is a scalar int
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

    def __iter__(self) -> Iterator[Tuple[T, hax.NamedArray]]:
        for item in self.dataset:
            yield item, self.domain_index


def _compute_per_domain_losses(Domain, domains, losses):
    one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
    per_domain_losses = hax.dot(losses.axes, one_hot_domains, losses)
    return per_domain_losses


def _domain_weighted_loss(losses, Domain, domains, alpha):
    one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
    return hax.dot(alpha, one_hot_domains, losses, axis=None)
