from typing import Iterator, TypeVar

import jax.lax
import jax.random as jrandom
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray


import haliax as hax
from haliax.types import IntScalar
from levanter.data import Dataset, ShardableDataset
from levanter.data.mixture import MixtureDataset
from levanter.trainer import Trainer, TrainerConfig

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

    def doremi_step(opt_state, proxy, alpha, batch, domains):
        # calculate per-elem losses for proxy and ref
        proxy_losses = TODO
        ref_losses = TODO
        # calculate excess losses
        excess_losses = hax.max(proxy_losses - ref_losses, 0)
        def total_losses_per_domain(excess_losses, domain):
            return jax.lax.cond(
                hax.sum(domains == domain) == 0,
                lambda: hax.zeros(()),
                lambda: hax.mean(excess_losses, where=domains == domain),
            )

        per_domain_losses = hax.vmap(total_losses_per_domain, Domain)(excess_losses, hax.arange(Domain))
        # Update domain weights (exp is entrywise): α ← α exp(ηλt)
        alpha = alpha * hax.exp(domain_weight_step * per_domain_losses)
        # Renormalize and smooth domain weights: α ← (1 − c) αPk i=1 α′ t[i] + cu
        alpha /= hax.sum(alpha)
        alpha = (1 - smoothing) * alpha + initial_alpha * smoothing
        # Update proxy model weights θt for the objective L(θt−1, αt) (using Adam, Adafactor, etc.)
        optimizer.update()






    def update_one_domain(per_token_losses, domains, target_domain):
        total_in_domain = jnp.sum(domains == target_domain)


    loader = trainer.sharded_loader(tagged_mixture, trainer.TrainBatch)
    for batch, tags in loader:
        state = trainer.train_step(state, batch)
        trainer.run_hooks(state)

        # Compute per-domain excess losses for each domain i ∈ {1, 2, ..., k} (ℓ_ο_j(x) is j-th token-level loss):
        # λ_t[i] ← (1 / |B\cap D_i|) * Σ_(x ∈ B\cap D_i) (1 / |x|) * Σ_(x ∈ BD_i) Σ_(j=1)^|x| max{ℓ_(t-1,j)(x) - ℓ_ref,j(x), 0}



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
            domain_index: int|hax.NamedArray,
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








