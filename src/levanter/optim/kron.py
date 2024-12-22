from dataclasses import dataclass
from typing import Optional, Union

import jax.numpy as jnp
import optax

from levanter.optim.config import OptimizerConfig


@OptimizerConfig.register_subclass("kron")
@dataclass
class KronConfig(OptimizerConfig):
    """Configuration for PSGD Kron optimizer.

    Attributes:
        beta1: Momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Optional gradient norm clipping value.
        normalize_grads: Whether to normalize the incoming gradients to unit norm layer-wise.
            Can help with stability but likely not necessary in this scenario.
        preconditioner_update_probability: Final probability of updating the preconditioner. Default
            is 0.05 (update every 20 steps). The `precond_update_prob_schedule` holds probability at
            1.0 for `update_prob_flat_start` steps before annealing exponentially down to this
            value within ~3000 steps. Training is slower while updates are done every step, but
            training speeds up after update probability decays.
        update_prob_flat_start: Number of steps to keep update probability at 1.0 before annealing.
            Default value of 500 works well, but increasing this to 1000 or 2000 can benefit training.
            However, this slows down training. A good balance is to keep update probability at 1.0 during
            initial loss drop, then when you notice loss start to plateau, the preconditioner is mostly
            learned and update probability can be decayed for faster training.
        max_size_triangular: Max size for dim's preconditioner to be triangular.
        min_ndim_triangular: Minimum number of dimensions a layer needs to have triangular preconditioners.
        memory_save_mode: Memory saving mode for preconditioners. Options:
            - None: All preconditioners are triangular (default)
            - 'one_diag': Largest/last dim per layer uses diagonal preconditioner
            - 'all_diag': All preconditioners are diagonal
        preconditioner_lr: Learning rate for preconditioner.
        preconditioner_init_scale: Scale for preconditioner initialization.
        mu_dtype: Dtype of the momentum buffer. Defaults to same dtype as parameters.
        precond_dtype: Dtype of the preconditioners. Defaults to 'float32'.
        precond_update_precision: Precision for matmul during preconditioner update.
            Options: 'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: Precision for matmul during preconditioning grads.
            Options: 'bfloat16', 'tensorfloat32', 'float32'.
        lax_map_scanned_layers: Whether to use lax.map for scanned layers instead of vmap.
            Useful to save memory with large models.
        lax_map_batch_size: Batch size for lax.map, see JAX docs for more info.
    """
    # some of these are changed from kron defaults to better suit levanter
    beta1: float = 0.9
    weight_decay: float = 0.1
    max_grad_norm: Optional[float] = 0.0
    normalize_grads: bool = False
    preconditioner_update_probability: float = 0.05
    update_prob_flat_start: int = 1000
    max_size_triangular: int = 25000
    min_ndim_triangular: int = 2
    memory_save_mode: Optional[str] = None
    preconditioner_lr: float = 0.1
    preconditioner_init_scale: float = 1.0
    mu_dtype: Optional[Union[str, jnp.dtype]] = None
    precond_dtype: Optional[Union[str, jnp.dtype]] = None
    precond_update_precision: Optional[str] = "tensorfloat32"
    precond_grads_precision: Optional[str] = None
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8

    def build(self, num_train_steps):
        """Creates the optimizer."""

        def _optimizer(learning_rate) -> optax.GradientTransformation:
            components = []
            if self.max_grad_norm and not self.normalize_grads:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(
                scale_by_kron_for_levanter(
                    b1=self.beta1,
                    normalize_grads=self.normalize_grads,
                    preconditioner_update_probability=precond_update_prob_schedule(
                        min_prob=self.preconditioner_update_probability,
                        flat_start=self.update_prob_flat_start,
                    ),
                    max_size_triangular=self.max_size_triangular,
                    min_ndim_triangular=self.min_ndim_triangular,
                    memory_save_mode=self.memory_save_mode,
                    preconditioner_lr=self.preconditioner_lr,
                    preconditioner_init_scale=self.preconditioner_init_scale,
                    mu_dtype=self.mu_dtype,
                    precond_dtype=self.precond_dtype,
                    precond_update_precision=self.precond_update_precision,
                    precond_grads_precision=self.precond_grads_precision,
                    lax_map_scanned_layers=self.lax_map_scanned_layers,
                    lax_map_batch_size=self.lax_map_batch_size,
                )
            )
            # PSGD's output should be RMS=1.0, so we can clip at 1.1 in case of incoming
            # gradient spike. This is better than clipping incoming grads because that would
            # get rid of valuable information for the preconditioner.
            components.append(optax.clip_by_block_rms(1.1))
            if self.weight_decay > 0:
                components.append(
                    optax.add_decayed_weights(
                        self.weight_decay, self.build_weight_decay_mask()
                    )
                )
            components.append(optax.scale_by_learning_rate(learning_rate))
            return optax.chain(*components)

        return optax.inject_hyperparams(_optimizer)(
            learning_rate=self.lr_scheduler(num_train_steps)
        )
    

"""PSGD Kron"""
from typing import Any, List, Optional, Union, Callable
from functools import partial
import string
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.lax import with_sharding_constraint
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain

import haliax as hax


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 250 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        return jnp.minimum(
            jnp.maximum(max_prob * jnp.exp(-decay * (n - flat_start)), min_prob),
            max_prob,
        )

    return _schedule


def scale_by_kron_for_levanter(
    b1: float = 0.9,
    normalize_grads: bool = False,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    momentum_into_precond_update: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
) -> base.GradientTransformation:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    A simple version of scale_by_kron that is focused on working only within levanter 
    with FSDP sharding for preconditioners.

    Args:
        b1: float, momentum parameter.
        normalize_grads: bool, whether to normalize gradients to unit norm layer-wise.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: bool, whether to send momentum into preconditioner
            update instead of raw gradients.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precond_update_precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
             'bfloat16', 'tensorfloat32', 'float32'.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.

    Returns:
        optax.GradientTransformation
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    def map_fn(do_map, fn, *args):
        """Maybe map a fn along first axis."""
        if do_map:
            if lax_map_scanned_layers:
                return jax.lax.map(
                    lambda xs: fn(*xs),
                    xs=args,
                    batch_size=lax_map_batch_size if lax_map_batch_size > 1 else None,
                )
            else:
                return vmap(fn)(*args)
        else:
            return fn(*args)

    def init_fn(params):
        def fsdp_size():
            mesh = hax.partitioning._get_mesh()
            fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
            fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
            fsdp_size = mesh.devices.shape[fsdp_axis]
            return fsdp_size

        # grab scanned layers and params sharding
        scanned_layers_ = jax.tree.map(
            lambda x: (
                jax.tree.map(lambda _: True, x, is_leaf=lambda x: isinstance(x, jax.Array))
                if isinstance(x, hax.nn.Stacked)
                else jax.tree.map(lambda _: False, x, is_leaf=lambda x: isinstance(x, jax.Array))
            ),
            params,
            is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
        )
        params_sharding_ = hax.partitioning.infer_resource_partitions(params)
        params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)

        params, params_structure = jax.tree.flatten(params, is_leaf=lambda x: isinstance(x, jax.Array))
        scanned_layers_ = params_structure.flatten_up_to(scanned_layers_)
        params_sharding_ = jax.tree.leaves(params_sharding_, is_leaf=lambda x: isinstance(x, PartitionSpec))
        # print(f"kron params: {jax.tree.map(lambda x: x.shape, params)}")
        # print(f"kron scanned_layers_: {scanned_layers_}")
        # print(f"kron params_sharding_: {params_sharding_}")

        # momentum
        mu = None
        if b1 > 0:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)
            mu = with_sharding_constraint(mu, params_sharding_)

        # preconditioners
        Qs = [
            _init_Q_exprs(
                t[0] if s else t,
                preconditioner_init_scale,
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
            )[0]
            for t, s in zip(jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]
        # broadcast for scanned layers
        def shard_q(q, s):
            q_shape_no_s = q.shape[int(s):]
            if len(q_shape_no_s) > 1 and q_shape_no_s[0] % fsdp_size() == 0:
                return with_sharding_constraint(
                    q, PartitionSpec(None, 'data') if s else PartitionSpec('data')
                )
            else:
                return with_sharding_constraint(q, PartitionSpec(None))

        Qs = [
            (
                jax.tree.map(
                    lambda d: shard_q(jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0), s), q
                )
                if s
                else q
            )
            for q, t, s in zip(
                Qs, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]

        # Calculate sizes for nu (preconditioner) and mu (momentum)
        Qs_n_elements = sum([q.size for q in jax.tree.leaves(Qs)])
        Qs_size_MB = sum(
            [q.size * q.dtype.itemsize / (2**20) for q in jax.tree.leaves(Qs)]
        )
        if jax.process_index() == 0:
            print(
                f"PSGD Preconditioners size: {Qs_n_elements} elements, "
                f"{Qs_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum([p.size for p in jax.tree.leaves(mu)])
            mu_size_MB = sum(
                [p.size * p.dtype.itemsize / (2**20) for p in jax.tree.leaves(mu)]
            )
            if jax.process_index() == 0:
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        # initial state
        return dict(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=jnp.zeros([], jnp.int32),
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key = jax.random.fold_in(jax.random.PRNGKey(42), state["count"])

        def fsdp_size():
            mesh = hax.partitioning._get_mesh()
            fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
            fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
            fsdp_size = mesh.devices.shape[fsdp_axis]
            return fsdp_size

        # grab scanned layers and params sharding
        scanned_layers_ = jax.tree.map(
            lambda x: (
                jax.tree.map(lambda _: True, x, is_leaf=lambda x: isinstance(x, jax.Array))
                if isinstance(x, hax.nn.Stacked)
                else jax.tree.map(lambda _: False, x, is_leaf=lambda x: isinstance(x, jax.Array))
            ),
            updates,
            is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
        )
        params_sharding_ = hax.partitioning.infer_resource_partitions(updates)
        params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)

        updates, grads_structure = jax.tree.flatten(updates, is_leaf=lambda x: isinstance(x, jax.Array))
        scanned_layers_ = grads_structure.flatten_up_to(scanned_layers_)
        params_sharding_ = jax.tree.leaves(params_sharding_, is_leaf=lambda x: isinstance(x, PartitionSpec))
        Qs = state["Qs_preconditioners"]
        # print(f"kron updates: {jax.tree.map(lambda x: x.shape, updates)}")
        # print(f"kron scanned_layers_: {scanned_layers_}")
        # print(f"kron params_sharding_: {params_sharding_}")

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        # normalize grads
        if normalize_grads:
            updates = jax.tree.map(
                lambda g: g / (jnp.linalg.norm(g) + 1e-16),
                updates,
            )

        # momentum
        mu = None
        momentum_updates = updates
        if state["mu"] is not None:
            mu = otu.tree_update_moment(updates, state["mu"], b1, 1)
            mu = with_sharding_constraint(mu, params_sharding_)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)
            momentum_updates = with_sharding_constraint(momentum_updates, params_sharding_)

        # get einsum expressions
        expressions = [
            _init_Q_exprs(
                t[0] if s else t,
                preconditioner_init_scale,
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
                precond_dtype,
                existing_Q=jax.tree.map(lambda d: d[0], Q) if s else Q,
            )
            for t, s, Q in zip(updates, scanned_layers_, Qs)
        ]

        # qs sharding
        def get_q_sharding(q, s):
            q_shape_no_s = q.shape[int(s):]
            if len(q_shape_no_s) > 1 and q_shape_no_s[0] % fsdp_size() == 0:
                return PartitionSpec(None, 'data') if s else PartitionSpec('data')
            else:
                return PartitionSpec(None)

        qs_sharding_ = [[get_q_sharding(q, s)for q in Q] for Q, s in zip(Qs, scanned_layers_)]

        # maybe update preconditioner
        def update_preconditioner(key, Qs):
            with jax.default_matmul_precision(precond_update_precision):
                if momentum_into_precond_update:
                    precond_updates_in = momentum_updates
                else:
                    precond_updates_in = updates

                # balance preconditioners about every 100 updates
                def balance_Qs(Qs: List[List[jax.Array]]):
                    def _balance_Q(Q: List[jax.Array]):
                        norms = jnp.array(
                            [jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32
                        )
                        gmean = jnp.prod(norms) ** (1 / len(norms))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return [
                        map_fn(s, _balance_Q, Q) if len(Q) > 1 else Q
                        for Q, s in zip(Qs, scanned_layers_)
                    ]

                key, subkey = jax.random.split(key)
                do_balances = jax.random.uniform(subkey) < 0.01
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)
                Qs = with_sharding_constraint(Qs, qs_sharding_)

                # create random vectors
                key, subkey = jax.random.split(key)
                Vs_keys = jax.random.split(subkey, len(precond_updates_in))
                Vs = [
                    jax.random.normal(k, shape=g.shape, dtype=g.dtype)
                    for k, g in zip(Vs_keys, precond_updates_in)
                ]
                Vs = with_sharding_constraint(Vs, params_sharding_)

                # damp based on machine precision (f32 probably enough)
                damp_eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
                precond_updates_in = jax.tree.map(
                    lambda g, v: g + damp_eps.astype(g.dtype) * jnp.mean(jnp.abs(g)) * v,
                    precond_updates_in,
                    Vs,
                )

                # form conjB
                conjBs = [
                    map_fn(s, _conjB, Q, g, v)
                    for s, Q, g, v in zip(scanned_layers_, Qs, precond_updates_in, Vs)
                ]
                conjBs = with_sharding_constraint(conjBs, params_sharding_)

                # update Qs
                new_Qs = [
                    map_fn(
                        s,
                        partial(
                            _update_precond, exprs=exprs, precond_lr=preconditioner_lr
                        ),
                        Q,
                        g,
                        conjb,
                    )
                    for s, exprs, Q, g, conjb in zip(
                        scanned_layers_, expressions, Qs, precond_updates_in, conjBs
                    )
                ]
                new_Qs = with_sharding_constraint(new_Qs, qs_sharding_)

                new_Qs = otu.tree_cast(new_Qs, precond_dtype)
                return new_Qs

        # update preconditioner deterministically
        update_counter_inc = safe_int32_increment(state["update_counter"])
        do_update = update_counter_inc >= 1 / update_prob_in
        update_counter_inc = jnp.where(do_update, 0, update_counter_inc)
        key, subkey = jax.random.split(key)
        Qs = jax.lax.cond(do_update, update_preconditioner, lambda _, qs: qs, subkey, Qs)
        Qs = with_sharding_constraint(Qs, qs_sharding_)

        # precondition gradients
        with jax.default_matmul_precision(precond_grads_precision):
            precond_gs = [
                map_fn(s, partial(_precond_grad, exprs=exprs), Q, g)
                for s, exprs, Q, g in zip(
                    scanned_layers_, expressions, Qs, momentum_updates
                )
            ]
            precond_gs = with_sharding_constraint(precond_gs, params_sharding_)

        # unflatten pytrees
        precond_gs = grads_structure.unflatten(precond_gs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=update_counter_inc,
        )

        return precond_gs, state

    return base.GradientTransformation(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    normalize_grads: bool = False,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    momentum_into_precond_update: bool = True,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
) -> base.GradientTransformation:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        b1: float, momentum parameter.
        normalize_grads: bool, whether to normalize gradients to unit norm layer-wise.
        weight_decay: float, weight decay.
        weight_decay_mask: optional Any or callable, pytree of bool same structure
            as params with weight decay applied to True elements.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular. 'one_diag' sets only the largest
            or last dim in a layer to be diagonal, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: bool, whether to send momentum into preconditioner
            update instead of raw gradients.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precond_update_precision: str, precision for matmul during preconditioner update,
            'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
            'bfloat16', 'tensorfloat32', 'float32'.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.

    Returns:
        optax.GradientTransformation
    """
    optimizer = [
        scale_by_kron_for_levanter(
            b1=b1,
            normalize_grads=normalize_grads,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precond_update_precision=precond_update_precision,
            precond_grads_precision=precond_grads_precision,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def _add_tiny(x):
    return x + jnp.finfo(x.dtype).tiny


def _norm_lower_bound(A: jax.Array):
    """Returns a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions and
    sizes suggest, norm(A) <= sqrt(2) * norm_lower_bound(A). Looks to be a very
    tight lower bound.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs
        aa = A * A

        aa_sum0 = jnp.sum(aa, axis=0)
        aa_sum1 = jnp.sum(aa, axis=1)
        i = jnp.argmax(aa_sum0, 0)
        j = jnp.argmax(aa_sum1, 0)
        value0 = jax.lax.dynamic_index_in_dim(aa_sum0, i, 0, keepdims=False)
        value1 = jax.lax.dynamic_index_in_dim(aa_sum1, j, 0, keepdims=False)

        def gt_branch():
            x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
            x = x @ A
            return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A.T)

        def le_branch():
            x = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
            x = A @ x
            return max_abs * jnp.linalg.norm(A.T @ (x / jnp.linalg.norm(x)))

        return jax.lax.cond(value0 > value1, gt_branch, le_branch)

    def no_calc(_):
        return max_abs

    return jax.lax.cond(max_abs > 0, calc, no_calc, A)


def _init_Q_exprs(
    t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype, existing_Q=None
):
    """For a scalar or tensor `t`, we initialize its preconditioner `Q` and
    reusable contraction expressions for updating `Q` and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = (
            [scale * jnp.ones_like(t, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'one_diag', 'all_diag']"
            )

        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.ones(size, dtype=dtype))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + 13])

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                if existing_Q is None:
                    def fsdp_size():
                        mesh = hax.partitioning._get_mesh()
                        fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
                        fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
                        fsdp_size = mesh.devices.shape[fsdp_axis]
                        return fsdp_size

                    new_q = scale * jnp.eye(size, dtype=dtype)
                    if new_q.shape[0] % fsdp_size() == 0:
                        new_q = with_sharding_constraint(new_q, PartitionSpec('data'))
                    Q.append(new_q)

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                exprGs.append(
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return exprA, exprGs, exprP
    return [Q, (exprA, exprGs, exprP)]


def _solve_triangular_right(X, A):
    """Compute X @ inv(A).

    A triangular solve has roughly the same complexity as a matmul.
    """
    X_ndim = X.ndim
    if X_ndim < 2:
        X = X[None, :]

    dtype_in = jnp.promote_types(A.dtype, X.dtype)
    A, X = A.astype(dtype_in), X.astype(dtype_in)
    leading_dims = 0
    if X.ndim > 2:
        leading_dims = X.ndim - 2
    solve_fn = partial(jax.lax.linalg.triangular_solve, left_side=False, lower=False)
    for _ in range(leading_dims):
        solve_fn = vmap(solve_fn, in_axes=(None, 0))
    solution = solve_fn(A, X)

    if X_ndim < 2:
        return solution[0]
    return solution


def _conjB(Q, G, V):
    """Compute conjB."""
    order = G.ndim
    p = list(range(order))
    conjB = jnp.transpose(V, p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.ndim < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = jnp.swapaxes(conjB, i, order - 1)
    return conjB


def _update_precond(Q, G, conjB, exprs, precond_lr):
    """Compute A and update Q."""
    exprA, exprGs, _ = exprs

    A = jnp.einsum(exprA, *Q, G)

    def _update_single_q(i, q):
        term1 = jnp.einsum(exprGs[i], A, A)
        term2 = jnp.einsum(exprGs[i], conjB, conjB)

        if q.ndim < 2:
            q -= (
                precond_lr
                / _add_tiny(jnp.max(jnp.abs(term1 + term2)))
                * (term1 - term2)
                * q
            )
        else:
            # main place I've found so far that needs specific sharding constraint is 
            # here on terms with transposed q sharding
            term1 = with_sharding_constraint(term1, PartitionSpec(None, 'data'))
            term2 = with_sharding_constraint(term2, PartitionSpec(None, 'data'))

            q -= (
                precond_lr
                / _add_tiny(_norm_lower_bound(term1 + term2))
                * jnp.triu(term1 - term2)
                @ q
            )
        return q

    return [_update_single_q(i, q) for i, q in enumerate(Q)]


def _precond_grad(Q, G, exprs):
    """Precondition gradient G with preconditioner Q."""
    exprP = exprs[-1]
    return jnp.einsum(exprP, *Q, *Q, G)
