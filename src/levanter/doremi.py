import dataclasses
import logging
from typing import Optional, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax.types import IntScalar

import levanter.tracker
from levanter.callbacks import eval_loss_loop
from levanter.checkpoint import load_checkpoint_or_initialize
from levanter.data import AsyncDataset, MappedAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.tracker import capture_time
from levanter.trainer import M, StepInfo, Trainer, TrainerConfig, TrainerState
from levanter.types import ComputeLossFunction
from levanter.utils.tree_utils import inference_mode


logger = logging.getLogger(__name__)


T = TypeVar("T")


# TODO: should we put ref in the state? If so, need to tell it to not serialize it
class DoremiState(TrainerState):
    alpha: hax.NamedArray
    average_alpha: hax.NamedArray

    def update_alpha(self, alpha):
        average_alpha = self.average_alpha + (alpha - self.average_alpha) / (self.step + 1)
        return dataclasses.replace(self, alpha=alpha, average_alpha=average_alpha)


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
    loss_fn: ComputeLossFunction[M, T],
    initial_proxy: M,
    ref: M,
    data_sources: dict[str, AsyncDataset[T]],
    sampling_weights: Optional[dict[str, float]] = None,
    *,
    validation_sets: Optional[dict[str, AsyncDataset[T]]] = None,
    trainer_config: TrainerConfig = DEFAULT_DOREMI_TRAINER_CONFIG,
    optimizer: optax.GradientTransformation = optax.adamw(1e-3),
    domain_weight_step_size: float = 1.0,
    smoothing: float = 1e-3,
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
    Domain = hax.Axis("domain", len(domain_indices))
    initial_alpha = hax.ones(Domain) / Domain.size

    trainer = Trainer(trainer_config, optimizer, loss_fn)
    with trainer:
        ref = _prepare_ref_model(ref, trainer_config)

        if validation_sets is not None:

            @eqx.filter_jit
            def eval_loss(model, *batch, **batch_kwargs):
                model = inference_mode(model, True)
                return trainer.loss_fn(model, *batch, **batch_kwargs, key=None)

            for domain, dataset in validation_sets.items():
                loss = eval_loss_loop(
                    eval_loss,
                    ref,
                    trainer.data_loader(dataset, trainer.EvalBatch),
                    name=f"ref {domain}",
                    max_batches=trainer_config.max_eval_batches,
                )
                print(f"Loss of ref model on domain {domain}: {loss:.3f}")
                levanter.tracker.log_metrics({f"eval/ref/{domain}/loss": loss}, step=0, commit=False)

        if validation_sets is not None:
            for domain, dataset in validation_sets.items():
                trainer.add_eval_hook(dataset, name=domain)

    if sampling_weights is not None:
        assert set(sampling_weights.keys()) == set(data_sources.keys())
        sampling_weights = {
            domain: weight / sum(sampling_weights.values()) for domain, weight in sampling_weights.items()
        }
    else:
        sampling_weights = {domain: 1 / len(data_sources) for domain in data_sources.keys()}

    # Loss is \sum_d alpha_d * (proxy - ref) (basically the unclipped excess loss with the new alpha)
    # Note that (\sum_d \alpha_d ref) is a constant in the model params, so we can ignore it for gradient computation
    # (JAX would ignore it for us I think but it's nice to be explicit and lets us log better)
    @hax.named_jit(axis_resources=trainer.parameter_axis_mapping, donate_args=(True,))
    def doremi_step(state: DoremiState, ref, batch, domains):
        proxy = inference_mode(state.model, False)
        with hax.axis_mapping(trainer.compute_axis_mapping):
            # calculate per-token losses for proxy and ref
            proxy_losses, proxy_loss_bwd = eqx.filter_vjp(lambda p: loss_fn(p, batch, reduction_axis=()), proxy)
            ref_losses = loss_fn(ref, batch, reduction_axis=())

            # calculate excess losses, aggregate per-domain losses
            excess_losses = proxy_losses - ref_losses
            clipped_losses = hax.maximum(excess_losses, 0)
            per_domain_losses = _compute_per_domain_losses(clipped_losses, Domain, domains)

            # Update domain weights
            alpha = state.alpha * hax.exp(domain_weight_step_size * per_domain_losses)
            alpha /= hax.sum(alpha)
            alpha = (1 - smoothing) * alpha + initial_alpha * smoothing

            # Update proxy model weights θt for the objective L(θt−1, αt) (using Adam, Adafactor, etc.)
            # Note DoReMi says to use the unclipped excess loss here. Confirmed with Michael
            loss, grad_loss = eqx.filter_value_and_grad(_domain_weighted_loss)(excess_losses, Domain, domains, alpha)
            grad = proxy_loss_bwd(grad_loss)[0]

        new_state = state.take_step(grad)
        new_state = new_state.update_alpha(alpha)

        # log metrics
        distance_from_uniform = hax.sum(hax.abs(alpha - initial_alpha))
        mean_excess_loss = hax.mean(excess_losses).scalar()
        mean_proxy_loss = hax.mean(proxy_losses).scalar()
        alpha_distance = hax.sum(hax.abs(new_state.average_alpha - state.average_alpha))
        alpha_dict = _decode_domain_array(Domain, new_state.average_alpha, domain_to_index)
        per_domain_dict = _decode_domain_array(Domain, per_domain_losses, domain_to_index)

        # log 0.0 as NaN so we can filter it out in the UI
        # need to use where b/c we're in jit
        per_domain_dict = {k: jnp.where(v == 0.0, jnp.nan, v) for k, v in per_domain_dict.items()}

        levanter.tracker.jit_log_metrics(
            {
                "change_in_alpha": alpha_distance.scalar(),
                "alpha_distance_from_uniform": distance_from_uniform.scalar(),
                "train/mean_excess_loss": mean_excess_loss,
                "train/mean_proxy_loss": mean_proxy_loss,
                **{f"alpha/{domain}": weight for domain, weight in alpha_dict.items()},
                # just skip domains with no excess loss
                **{f"train/{domain}/excess_loss": loss for domain, loss in per_domain_dict.items()},
            },
            step=state.step,
        )

        return loss, alpha_distance, new_state

    # we're not actually going to use the trainer for very much but it holds hooks and sets up contexts
    with trainer:
        tagged_mixture = domain_tagged_mixture(data_sources, sampling_weights, domain_to_index, key=data_key)
        state = load_checkpoint_or_initialize(
            DoremiState.init,
            trainer.checkpoint_path,
            axis_mapping=trainer.parameter_axis_mapping,
            mesh=trainer.device_mesh,
            donate_kwargs=None,
        )(
            trainer.optimizer,
            initial_proxy,
            key=training_key,
            is_trainable=True,
            mp=trainer.mp,
            alpha=initial_alpha,
            average_alpha=initial_alpha,
        )
        del initial_proxy
        train_loader = iter(trainer.data_loader(tagged_mixture, trainer.TrainBatch))

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

        trainer.run_hooks(new_info, force=True)

        alpha = state.average_alpha
        final_weights = _decode_domain_array(Domain, alpha, domain_to_index)

        levanter.tracker.log_summary({"final_alpha": final_weights})

    return {k: float(v) for k, v in final_weights.items()}


def _decode_domain_array(Domain, alpha, domain_name_to_index):
    final_weights = {domain: alpha[Domain, index].scalar() for domain, index in domain_name_to_index.items()}
    return final_weights


def _compute_per_domain_losses(losses, Domain, domains):
    """Compute per-domain average losses from a batch of losses"""
    # out[d] = E[losses | domain=d]
    one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
    per_domain_losses = hax.dot(one_hot_domains, losses, axis=losses.axes, out_axes=(Domain,))
    # count the number of losses for each domain
    norm = hax.dot(one_hot_domains, losses != 0, axis=losses.axes, out_axes=(Domain,))
    norm = hax.maximum(norm, 1.0)  # don't nan if there are no losses for a domain
    return per_domain_losses / norm


def _domain_weighted_loss(losses, Domain, domains, alpha):
    """Average loss weighted by domain weights"""
    per_domain_losses = _compute_per_domain_losses(losses, Domain, domains)
    return hax.dot(alpha, per_domain_losses, axis=Domain).scalar()


def _prepare_ref_model(ref, trainer):
    return hax.named_jit(
        lambda m: trainer.mp.cast_to_compute(inference_mode(m, True)),
        axis_resources=trainer.parameter_axis_mapping,
        donate_args=True,
    )(ref)


def domain_tagged_mixture(
    data_sources: dict[str, AsyncDataset[T]],
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

    return MixtureDataset(tagged_datasets, weights, key=key, block_size=2048)


class DomainTaggedDataset(MappedAsyncDataset[T, Tuple[T, hax.NamedArray]]):  # named array is a scalar int
    def __init__(
        self,
        dataset: AsyncDataset[T],
        domain_index: int | hax.NamedArray,
    ):
        self.dataset = dataset

        if isinstance(domain_index, int):
            self.domain_index = hax.named(jnp.array(domain_index, dtype=int), ())
        else:
            self.domain_index = domain_index

        def _transform(item):
            return item, self.domain_index

        super().__init__(dataset, _transform)
