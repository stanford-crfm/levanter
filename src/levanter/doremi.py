from typing import Iterator, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax.types import IntScalar

from levanter.data import Dataset, ShardableDataset
from levanter.data.mixture import MixtureDataset


T = TypeVar("T")


def estimate_mixture_weights(
    optimizer: optax.GradientTransformation,
    loss_fn,
    initial_proxy,
    ref,
    data_sources: dict[str, ShardableDataset[T]],
    ref_weights: dict[str, float],
    domain_weight_step: float = 1.0,
    smoothing: float = 1e-3,
    *,
    key: PRNGKeyArray,
) -> dict[str, float]:
    """
    Estimate the mixture weights for the data sources using DoReMi.
    https://arxiv.org/abs/2305.10429
    """
    training_key, data_key = jrandom.split(key)
    domain_indices = list(data_sources.keys())
    domain_to_index = {domain: index for index, domain in enumerate(domain_indices)}
    tagged_mixture = domain_tagged_mixture(data_sources, ref_weights, domain_to_index, key=data_key)

    state = trainer.initial_state(training_key, model=initial_proxy)

    del initial_proxy

    # Initialize domain weights
    Domain = hax.Axis("domain", len(domain_indices))
    initial_alpha = hax.ones(Domain) / Domain.size

    # calculate per-token losses for proxy and ref
    def compute_excess_loss(proxy, ref, batch):
        proxy_losses = TODO
        ref_losses = TODO
        # calculate excess losses
        excess_losses = proxy_losses - ref_losses
        return excess_losses

    # Loss is alpha_d * (proxy - ref) (basically the unclipped excess loss with the new alpha)
    def proxy_model_loss(excess_losses, domains, alpha):
        one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
        # basically einsum(" * -> ", alpha, one_hot_domains, excess_losses)
        loss = hax.dot(excess_losses.axes + (Domain,), alpha, one_hot_domains, excess_losses).scalar()

        return loss

    def doremi_step(opt_state, proxy, alpha, batch, domains):
        # this is one of those times when PyTorch's backward() is nice
        excess_losses, excess_backward = eqx.filter_vjp(lambda proxy: compute_excess_loss(proxy, ref, batch), proxy)

        # Update domain weights
        ## Compute per-domain excess losses
        clipped_losses = hax.maximum(excess_losses, 0)
        one_hot_domains = hax.nn.one_hot(domains, Domain)  # Domain x Batch
        per_domain_losses = hax.dot(excess_losses.axes, one_hot_domains, clipped_losses)

        old_alpha = alpha
        alpha = alpha * hax.exp(domain_weight_step * per_domain_losses)
        alpha /= hax.sum(alpha)
        alpha = (1 - smoothing) * alpha + initial_alpha * smoothing

        alpha_distance = hax.sum(hax.abs(alpha - old_alpha))

        # Update proxy model weights θt for the objective L(θt−1, αt) (using Adam, Adafactor, etc.)
        val, grad_loss = eqx.filter_value_and_grad(proxy_model_loss)(excess_losses, domains, alpha)
        grad = excess_backward(grad_loss)

        updates, new_state = optimizer.update(opt_state, grad, params=proxy)
        proxy = optax.apply_updates(proxy, updates)

        return new_state, proxy, alpha


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
