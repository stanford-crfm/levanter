import string
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic, List, Optional, Tuple, TypeVar, Union, cast

import chex
import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax import vmap
from jax.lax import with_sharding_constraint
from jax.sharding import PartitionSpec
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.combine import chain
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype

from levanter.optim.config import OptimizerConfig


# Define type variables for the pytree structure
T = TypeVar("T")
PartitionSpecTree = TypeVar(
    "PartitionSpecTree", bound=Union[PartitionSpec, List[PartitionSpec], Tuple[PartitionSpec, ...], dict, list, tuple]
)


@OptimizerConfig.register_subclass("quad")
@dataclass
class QUADConfig(OptimizerConfig, Generic[PartitionSpecTree]):
    """Configuration for PSGD-QUAD optimizer.

    Notes:
        - LR is usually 3x smaller than adam, weight decay at least 3x larger.
        - use with ~100 steps of LR warmup.
        - max_skew_dense default is 1.0 making larger dimensions have diagonal preconditioners, but this 
          can be set to float('inf') to make all preconditioners dense (more memory use but might be more stable).
          For example, for layer shape (256, 128) default preconditioners will be (256,) and (128, 128), but if 
          max_skew_dense is set to float('inf'), then preconditioners will be (256, 256) and (128, 128).

    Attributes:
        beta1: Momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Optional gradient norm clipping value.
        normalize_grads: Whether to normalize the incoming gradients to unit norm layer-wise.
            Can help with stability.
        max_size_dense: Dimensions larger than this will have diagonal preconditioners,
            otherwise dense.
        max_skew_dense: Dimensions with skew larger than this compared to the other
            dimension will have diagonal preconditioners, otherwise dense.
        preconditioner_lr: Learning rate for preconditioner.
        preconditioner_init_scale: Scale for preconditioner initialization.
        mu_dtype: Dtype of the momentum buffer. Defaults to same dtype as parameters.
        precond_dtype: Dtype of the preconditioners. Defaults to 'float32'.
        scanned_layers: Tree of booleans same structure as params indicating scanned dimensions
            for each layer. PSGD will vmap over leading dimension.
        lax_map_scanned_layers: Whether to use lax.map for scanned layers instead of vmap.
            Useful to save memory with large models.
        lax_map_batch_size: Batch size for lax.map, see JAX docs for more info.
        merge_small_dims: Whether to merge small dimensions to improve preconditioner efficiency.
        target_merged_dim_size: Target size of merged dimensions.
        partition_grads_into_blocks: Whether to partition grads into chunks of size block_size
            for efficiency.
        block_size: Block size to use for partitioning grads.
        params_sharding: Pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: PartitionSpec for preconditioner matrices. Best practice is to
            shard first dimension across fsdp-like mesh axis, or largest/most common axis in params.
            Example: PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').
    """

    # some of these are changed from quad defaults to better suit levanter
    beta1: float = 0.95
    weight_decay: float = 0.3
    max_grad_norm: Optional[float] = 1.0
    normalize_grads: bool = False
    max_size_dense: int = 8192
    max_skew_dense: float = 1.0
    preconditioner_lr: float = 0.9
    preconditioner_init_scale: float = 1.0
    mu_dtype: Optional[Union[str, jnp.dtype]] = None
    precond_dtype: Optional[Union[str, jnp.dtype]] = None
    lax_map_scanned_layers: bool = False
    lax_map_batch_size: int = 8
    merge_small_dims: bool = True
    target_merged_dim_size: int = 8192
    partition_grads_into_blocks: bool = False
    block_size: int = 512
    params_sharding: Optional[PartitionSpecTree] = None
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None

    def build(self, num_train_steps):
        """Creates the optimizer."""

        def _optimizer(learning_rate) -> optax.GradientTransformation:
            precond_partition_spec = (
                PartitionSpec(*self.preconditioner_sharding) if self.preconditioner_sharding is not None else None
            )
            components = []
            if self.max_grad_norm and not self.normalize_grads:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(
                scale_by_quad(
                    b1=self.beta1,
                    normalize_grads=self.normalize_grads,
                    max_size_dense=self.max_size_dense,
                    max_skew_dense=self.max_skew_dense,
                    preconditioner_lr=self.preconditioner_lr,
                    preconditioner_init_scale=self.preconditioner_init_scale,
                    mu_dtype=self.mu_dtype,
                    precond_dtype=self.precond_dtype,
                    lax_map_scanned_layers=self.lax_map_scanned_layers,
                    lax_map_batch_size=self.lax_map_batch_size,
                    merge_small_dims=self.merge_small_dims,
                    target_merged_dim_size=self.target_merged_dim_size,
                    partition_grads_into_blocks=self.partition_grads_into_blocks,
                    block_size=self.block_size,
                    params_sharding=self.params_sharding,
                    preconditioner_sharding=precond_partition_spec,
                )
            )
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            components.append(optax.scale_by_learning_rate(learning_rate))
            return optax.chain(*components)

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


"""PSGD-QUAD"""

try:
    import flax.linen as nn

    have_flax = True
except ImportError:
    have_flax = False
try:
    import haliax as hax

    have_hax = True
except ImportError:
    have_hax = False


def scale_by_quad(
    b1: float = 0.95,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    max_skew_dense: float = 1.0,
    preconditioner_lr: float = 0.9,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = True,
    target_merged_dim_size: int = 8192,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[PartitionSpecTree] = None,
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None,
    **kwargs,
) -> base.GradientTransformation:
    """
    Implements PSGD-QUAD from https://github.com/lixilinx/psgd_torch.
    Author: https://github.com/evanatyourservice

            Args:
        b1: float, momentum parameter. 0.9 or 0.95 are common values.
        normalize_grads: bool, whether to normalize the incoming gradients to unit
            norm layer-wise. Can help with stability.
        max_size_dense: int, dimensions larger than this will have diagonal preconditioners,
            otherwise dense.
        max_skew_dense: float, dimensions with skew larger than this compared to the other
            dimension will have diagonal preconditioners, otherwise dense.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum buffer. Defaults to
            same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioners. Defaults
            to 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge small dimensions to improve
            preconditioner efficiency.
        target_merged_dim_size: int, target size of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for efficiency.
        block_size: int, block size to use for partitioning grads.
        params_sharding: pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: `None` or `PartitionSpec(str | None, str | None)`,
            PartitionSpec for preconditioner matrices. `None` infers a strategy
            from params_sharding that matches first preconditioner axis to
            corresponding axis in params. Best practice, though, is to shard the first
            dimension across fsdp-like mesh axis, or the largest, most common axis in
            params. For example, PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').

    Returns:
        optax.GradientTransformation
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype or jnp.float32)
    lax_map = lax_map_scanned_layers
    bs = lax_map_batch_size
    scanned_layers = None

    def init_fn(params, return_partition_specs_only=False):
        # unbox if haliax style partitioned
        scanned_layers_ = None
        params_sharding_ = params_sharding
        if have_hax:
            if any(
                isinstance(x, hax.NamedArray)
                for x in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, hax.NamedArray))
            ):
                # if in haliax, we can grab scanned_layers and params_sharding from params
                # this does not support nested stacks
                if scanned_layers_ is None:
                    scanned_layers_ = jax.tree.map(
                        lambda x: (jax.tree.map(lambda _: True, x) if isinstance(x, hax.nn.Stacked) else False),
                        params,
                        is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
                    )
                if params_sharding_ is None:
                    params_sharding_ = hax.partitioning.infer_resource_partitions(params)
                    params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
                params, params_struct = jax.tree.flatten(params)
                scanned_layers_ = jax.tree.leaves(scanned_layers_)
                params_sharding_ = jax.tree.leaves(params_sharding_)

        have_params_sharding = params_sharding_ is not None
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None or have_hax

        # unbox if flax style partitioned
        if have_flax:
            params = jax.tree.map(
                lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
                params,
                is_leaf=lambda x: isinstance(x, nn.Partitioned),
            )

        # check that there is a PartitionSpec for every param
        if params_sharding_ is not None:
            assert len(jax.tree.leaves(params_sharding_)) == len(
                jax.tree.leaves(params)
            ), "There must be a PartitionSpec for every parameter in PSGD-QUAD."
        # check that preconditioner sharding length is at least 1
        if preconditioner_sharding is not None:
            assert len(preconditioner_sharding) > 0, (
                "preconditioner_sharding must have length > 0. For example, "
                "PartitionSpec(None) or PartitionSpec('fsdp', None) are valid."
            )

        # extend partition specs
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda p, sh: PartitionSpec(*(sh + (None,) * (len(p.shape) - len(sh)))),
                params,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(preconditioner_sharding[0], None)

        # reshape params shaped () to (1,) to make things simpler
        params = jax.tree.map(lambda p: p[None] if len(p.shape) == 0 else p, params)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)
        scanned_sizes = jax.tree.map(lambda p, s: p.shape[0] if s else 0, params, scanned_layers_)

        # momentum
        mu = None
        mu_sharding = params_sharding_
        if b1 > 0 and not return_partition_specs_only:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)
            # apply params sharding to momentum buffer
            if have_params_sharding:
                mu = _safe_sharding_constraint(mu, params_sharding_)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda p, s: _get_preconditioner_types(
                p.shape[int(s) :],
                max_size_dense,
                max_skew_dense,
            ),
            params,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, params)
        merged_shapes = jax.tree.map(lambda p, s: p.shape[int(s) :], params, scanned_layers_)
        if merge_small_dims:
            output = jax.tree.map(
                lambda p, s, dd, sh: _merge_small_dims(p.shape[int(s) :], target_merged_dim_size, dd, sh),
                params,
                scanned_layers_,
                dim_diag,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(3)
            ]

        # partition grads into blocks
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                params,
                merged_shapes,
                dim_diag,
            )
            # we can grab resulting shapes from partitioners
            partitioned_shapes = jax.tree.map(lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners)

        # initialize preconditioners
        output = jax.tree.map(
            lambda _, ps, dd, sh: list(
                _init_Q_exprs(
                    ps[1:] if partition_grads_into_blocks else ps,
                    preconditioner_init_scale,
                    dd,
                    precond_dtype,
                    existing_Q=True if return_partition_specs_only else None,
                    precond_sharding=preconditioner_sharding_,
                    param_sharding=sh,
                )
            ),
            params,
            partitioned_shapes,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
        )
        if return_partition_specs_only:
            exprs, Qs_sharding_no_leading_dims = [jax.tree.map(lambda _, x: x[i], params, output) for i in range(2)]
        else:
            Qs, Ls, exprs, Qs_sharding_no_leading_dims = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(4)
            ]
        Qs_sharding = None
        if have_qs_sharding:
            # add scan and stack dims to Qs sharding
            def add_dims_to_spec(_, qss, sds):
                if partition_grads_into_blocks:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
                if sds is not None:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
                return qss

            Qs_sharding = jax.tree.map(
                add_dims_to_spec,
                params,
                Qs_sharding_no_leading_dims,
                scanned_dim_sharding,
            )

        if not return_partition_specs_only:
            # broadcast Qs and Ls for stacks and scans
            def broadcast_qs(_, ps, x, s):
                stack_n = ps[0]
                if partition_grads_into_blocks:
                    # add leading dim for stacked partitions
                    x = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), stack_n, axis=0), x)
                if s > 0:
                    # add leading dim if we're scanning this layer
                    x = jax.tree.map(lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), x)
                return x

            Qs = jax.tree.map(broadcast_qs, params, partitioned_shapes, Qs, scanned_sizes)
            Ls = jax.tree.map(broadcast_qs, params, partitioned_shapes, Ls, scanned_sizes)
            if have_qs_sharding:
                Qs = _safe_sharding_constraint(Qs, Qs_sharding)

        if return_partition_specs_only:
            return dict(
                count=PartitionSpec(),
                mu=mu_sharding,
                Qs_preconditioners=Qs_sharding,
                Ls_lipschitz=PartitionSpec(None),
            )

        return dict(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])

        # unbox if haliax style partitioned
        scanned_layers_ = scanned_layers
        params_sharding_ = params_sharding
        hax_partitioned = False
        if have_hax:
            if any(
                isinstance(x, hax.NamedArray)
                for x in jax.tree.leaves(updates, is_leaf=lambda x: isinstance(x, hax.NamedArray))
            ):
                hax_partitioned = True
                # if in haliax, we can grab scanned_layers and params_sharding from params
                # this does not support nested stacks
                if scanned_layers_ is None:
                    scanned_layers_ = jax.tree.map(
                        lambda x: (jax.tree.map(lambda _: True, x) if isinstance(x, hax.nn.Stacked) else False),
                        updates,
                        is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
                    )
                if params_sharding_ is None:
                    params_sharding_ = hax.partitioning.infer_resource_partitions(updates)
                    params_sharding_ = jax.tree.map(lambda x: x.spec, params_sharding_)
                updates, updates_struct = jax.tree.flatten(updates)
                scanned_layers_ = jax.tree.leaves(scanned_layers_)
                params_sharding_ = jax.tree.leaves(params_sharding_)

        have_params_sharding = params_sharding_ is not None
        if have_params_sharding:
            original_params_sharding_ = params_sharding_
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None or have_hax

        # unbox if flax style partitioned
        flax_partitioned = False
        if have_flax:
            boxed_updates, grads_structure = jax.tree.flatten(
                updates,
                is_leaf=lambda g: isinstance(g, (chex.Array, nn.Partitioned, jax.ShapeDtypeStruct)),
            )
            if any(isinstance(g, nn.Partitioned) for g in boxed_updates):
                flax_partitioned = True
                updates = [g.unbox() for g in boxed_updates]
                updates = grads_structure.unflatten(updates)

        # extend partition specs
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda g, sh: PartitionSpec(*(sh + (None,) * (len(g.shape) - len(sh)))),
                updates,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(preconditioner_sharding[0], None)

        # reshape params shaped () to (1,) to make things simpler
        input_shapes = jax.tree.map(lambda g: g.shape, updates)
        updates = jax.tree.map(lambda g: g[None] if len(g.shape) == 0 else g, updates)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        if scanned_layers_ is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        # normalize grads
        def norm_grads(g):
            return g / (jnp.linalg.norm(g) + 1e-6)

        if normalize_grads:
            updates = jax.tree.map(norm_grads, updates)

        # momentum
        mu = None
        momentum_updates = updates
        if state["mu"] is not None:
            mu = otu.tree_update_moment(updates, state["mu"], b1, 1)
            if have_params_sharding:
                mu = _safe_sharding_constraint(mu, params_sharding_)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda g, s: _get_preconditioner_types(
                g.shape[int(s) :],
                max_size_dense,
                max_skew_dense,
            ),
            momentum_updates,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, momentum_updates)
        merged_params_sharding = params_sharding_
        original_shapes = None
        if merge_small_dims:
            original_shapes = jax.tree.map(lambda g, s: g.shape[int(s) :], momentum_updates, scanned_layers_)
            output = jax.tree.map(
                lambda g, dd, s, sh: _merge_small_dims(g.shape[int(s) :], target_merged_dim_size, dd, sh),
                momentum_updates,
                dim_diag,
                scanned_layers_,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], momentum_updates, output) for i in range(3)
            ]
            # reshape
            momentum_updates = jax.tree.map(
                lambda g, s, ns: _map_fn(False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g),
                momentum_updates,
                scanned_layers_,
                merged_shapes,
            )
            if have_params_sharding:
                # scanned dim sharding + new merged sharding
                merged_params_sharding = jax.tree.map(
                    lambda sws, sds: PartitionSpec(*(sds + sws if sds is not None else sws)),
                    sharding_without_scan,
                    scanned_dim_sharding,
                )
        # constrain sharding
        if have_params_sharding:
            momentum_updates = _safe_sharding_constraint(momentum_updates, merged_params_sharding)

        # partition grads into blocks
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        partitioned_sharding = merged_params_sharding
        partitioners = None
        partitioned_shapes = None
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda g, dd, s: BlockPartitioner(g.shape[int(s) :], block_size, dd),
                momentum_updates,
                dim_diag,
                scanned_layers_,
            )
            # layers become tuples each containing layer's partitions
            momentum_updates = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                momentum_updates,
                partitioners,
                scanned_layers_,
            )
            partitioned_shapes = jax.tree.map(
                lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # constrain partitions to same sharding as entire layer
                momentum_updates = jax.tree.map(
                    lambda _, g, mps: jax.tree.map(lambda x: _safe_sharding_constraint(x, mps), g),
                    dummy_updates_tree,
                    momentum_updates,
                    merged_params_sharding,
                )
            # pad and stack partitions, tuples become arrays with new leading dim
            momentum_updates = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # add dim to sharding specs for new stacked dim
                partitioned_sharding = jax.tree.map(
                    lambda mps, s: PartitionSpec(*(mps[: int(s)] + (None,) + mps[1:])),
                    merged_params_sharding,
                    scanned_layers_,
                )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)
        # constrain sharding
        if have_params_sharding:
            momentum_updates = _safe_sharding_constraint(momentum_updates, partitioned_sharding)

        # get einsum expressions and Qs sharding
        Qs = state["Qs_preconditioners"]
        Ls = state["Ls_lipschitz"]
        Qs_sharding = None
        exprs_and_sharding = jax.tree.map(
            lambda g, dd, sh, nm: _init_Q_exprs(
                g.shape[nm:],
                preconditioner_init_scale,
                dd,
                precond_dtype,
                existing_Q=True,
                existing_L=True,
                precond_sharding=preconditioner_sharding_,
                param_sharding=sh,
            ),
            momentum_updates,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
            n_dims_to_map,
        )
        exprs, Qs_sharding_no_leading_dims = [
            jax.tree.map(lambda _, x: x[i], dummy_updates_tree, exprs_and_sharding) for i in range(2)
        ]
        Qs_sharding = None
        if have_qs_sharding:
            # add scan and stack dims to Qs sharding
            def add_dims_to_spec(_, qss, sds):
                if partition_grads_into_blocks:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
                if sds is not None:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
                return qss

            Qs_sharding = jax.tree.map(
                add_dims_to_spec,
                dummy_updates_tree,
                Qs_sharding_no_leading_dims,
                scanned_dim_sharding,
            )

        # balance preconditioners about every 100 updates
        def balance_Qs(Qs_to_bal):
            def _balance_Q(Q):
                norms = jnp.array([jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32)
                gmean = jnp.exp(jnp.mean(jnp.log(norms)))
                to_mul = gmean / norms
                return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

            return jax.tree.map(
                lambda _, Q, nm: _map_fn(False, 0, nm, _balance_Q, Q),
                dummy_updates_tree,
                Qs_to_bal,
                n_dims_to_map,
            )

        Qs = jax.lax.cond(count_inc % 100 == 0, balance_Qs, lambda qs: qs, Qs)
        if have_qs_sharding:
            Qs = _safe_sharding_constraint(Qs, Qs_sharding)

            # update Qs and constrain sharding
        with jax.default_matmul_precision("high"):
            Qs_Ls_Pg = jax.tree.map(
                lambda g, Q, L, expr, nm, qss, sh: _map_fn(
                    lax_map,
                    bs,
                    nm,
                    partial(
                        _update_precond,
                        exprs=expr,
                        precond_lr=preconditioner_lr,
                        qs_sharding=qss,
                        params_sharding=sh,
                    ),
                    Q,
                    L,
                    g,
                ),
                momentum_updates,
                Qs,
                Ls,
                exprs,
                n_dims_to_map,
                Qs_sharding_no_leading_dims if have_qs_sharding else nones,
                sharding_without_scan if have_params_sharding else nones,
            )
        Qs, Ls, precond_gs = [
            jax.tree_util.tree_map(lambda qlp: qlp[i], Qs_Ls_Pg, is_leaf=lambda x: isinstance(x, tuple))
            for i in range(3)
        ]
        if have_qs_sharding:
            Qs = _safe_sharding_constraint(Qs, Qs_sharding)
        if have_params_sharding:
            precond_gs = _safe_sharding_constraint(precond_gs, partitioned_sharding)
        Qs = otu.tree_cast(Qs, precond_dtype)

        # unpartition grads
        if partition_grads_into_blocks:
            precond_gs = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                precond_gs,
                scanned_layers_,
                partitioned_shapes,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, merged_params_sharding)
            precond_gs = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(False, 0, int(s), p_cls.merge_partitions, g),
                dummy_updates_tree,
                precond_gs,
                scanned_layers_,
                partitioners,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, merged_params_sharding)

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                precond_gs,
                scanned_layers_,
                original_shapes,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, params_sharding_)

        # return scalars to original shape
        precond_gs = jax.tree.map(lambda g, s: jnp.reshape(g, s), precond_gs, input_shapes)

        # final constraint for good measure
        if have_params_sharding:
            precond_gs = _safe_sharding_constraint(precond_gs, original_params_sharding_)

        # box preconditioned grads
        if flax_partitioned:
            flat_precond_gs, _ = jax.tree.flatten(precond_gs)
            precond_gs = [bu.replace_boxed(g) for bu, g in zip(boxed_updates, flat_precond_gs)]
            precond_gs = grads_structure.unflatten(precond_gs)
        if hax_partitioned:
            precond_gs = updates_struct.unflatten(precond_gs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs,
            Ls_lipschitz=Ls,
        )

        return precond_gs, state

    return base.GradientTransformation(init_fn, update_fn)


def quad(
    learning_rate: Union[float, Callable[[int], float]] = 0.0003,
    b1: float = 0.95,
    weight_decay: float = 0.3,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    normalize_grads: bool = False,
    max_size_dense: int = 8192,
    max_skew_dense: float = 1.0,
    preconditioner_lr: float = 0.9,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = True,
    target_merged_dim_size: int = 8192,
    partition_grads_into_blocks: bool = False,
    block_size: int = 512,
    params_sharding: Optional[PartitionSpecTree] = None,
    preconditioner_sharding: Optional[tuple[str | None, str | None]] = None,
) -> base.GradientTransformation:
    """
    Implements PSGD-QUAD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate schedule.
        b1: float, momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: float, weight decay coefficient.
        weight_decay_mask: optional pytree same structure as params, or callable
            returning a pytree, that masks weight decay. Weight decay is applied to
            leaves that are True.
        normalize_grads: bool, whether to normalize the incoming gradients to unit
            norm layer-wise. Can help with stability.

        max_size_dense: int, dimensions larger than this will have diagonal preconditioners,
            otherwise dense.
        max_skew_dense: float, dimensions with skew larger than this compared to the other
            dimension will have diagonal preconditioners, otherwise dense.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum buffer. Defaults to
            same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioners. Defaults
            to 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge small dimensions to improve
            preconditioner efficiency.
        target_merged_dim_size: int, target size of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for efficiency.
        block_size: int, block size to use for partitioning grads.
        params_sharding: pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: `None` or `PartitionSpec(str | None, str | None)`,
            PartitionSpec for preconditioner matrices. `None` infers a strategy
            from params_sharding that matches first preconditioner axis to
            corresponding axis in params. Best practice, though, is to shard the first
            dimension across fsdp-like mesh axis, or the largest, most common axis in
            params. For example, PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').

    Returns:
        optax.GradientTransformation
    """
    optimizer = [
        scale_by_quad(
            b1=b1,
            normalize_grads=normalize_grads,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            merge_small_dims=merge_small_dims,
            target_merged_dim_size=target_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            params_sharding=params_sharding,
            preconditioner_sharding=preconditioner_sharding,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def get_opt_state_partition_specs(params: base.Params, scale_by_quad_only: bool = False, **kwargs):
    """Get tree of PartitionSpecs for quad optimizer state.

    params converted to jax.ShapeDtypeStructs, no arrays are used.

    Args:
        params: pytree of Arrays, nn.Partitioned, or jax.ShapeDtypeStruct.
        scale_by_quad_only: bool, If True, only returns partition specs for the
            `scale_by_quad` function, otherwise the `quad` function.
        kwargs: kwargs for quad (or scale_by_quad).

    Returns:
        tree of PartitionSpecs for optimizer state.
    """
    params_flat, params_struct = jax.tree.flatten(params)
    if have_flax:
        if isinstance(params_flat[0], nn.Partitioned):
            params_flat = [p.unbox(p) for p in params_flat]
    if not isinstance(params_flat[0], jax.ShapeDtypeStruct):
        params_flat = [jax.ShapeDtypeStruct(p.shape, p.dtype) for p in params_flat]
    params = params_struct.unflatten(params_flat)

    specs = scale_by_quad(**kwargs).init(params, return_partition_specs_only=True)

    if not scale_by_quad_only:
        specs = (specs,)
        if kwargs.get("weight_decay", 0.0) > 0.0:
            specs += (None,)
        specs += (None,)

    return specs


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_size: int, max_skew: float
) -> List[bool]:
    if len(shape) == 0:
        return [True]
    
    total_numel = np.prod(shape)
    dim_diag = []
    for i, size in enumerate(shape):
        if size == 1 or len(shape) == 1 or size > max_size or size**2 > max_skew * total_numel:
            dim_diag.append(True)
        else:
            dim_diag.append(False)
    
    return dim_diag


def _init_Q_exprs(
    t_shape,
    scale,
    dim_diag,
    dtype,
    existing_Q=None,
    existing_L=None,
    precond_sharding=None,
    param_sharding=None,
):
    have_qs_sharding = precond_sharding is not None or param_sharding is not None
    letters = string.ascii_lowercase + string.ascii_uppercase

    if len(t_shape) > 13:
        raise ValueError(f"Got tensor with dim {len(t_shape.shape)}; Einstein runs out of letters!")
    scale = scale ** (1 / len(t_shape))
    Q = [] if existing_Q is None else existing_Q
    L = [] if existing_L is None else existing_L
    exprP = ",".join(letters[i + 13] for i in range(len(t_shape))) + "," + ",".join(letters[i + 26] for i in range(len(t_shape))) + "->" + ",".join(letters[i + 13] for i in range(len(t_shape)))
    piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
    exprGs = []

    params_specs = param_sharding
    if param_sharding is None:
        params_specs = PartitionSpec(*((None,) * len(t_shape)))
    sharding_out = [None] * len(t_shape)
    if have_qs_sharding:
        sharding_out = [PartitionSpec(None)] * len(t_shape)

    for i, (size, dim_d, dim_sh) in enumerate(zip(t_shape, dim_diag, params_specs)):
        if existing_L is None:
            L.append(jnp.zeros((1,), dtype=jnp.float32))
            
        if dim_d:
            # use diagonal matrix as preconditioner for this dim
            if existing_Q is None:
                q = scale * jnp.ones(size, dtype=dtype)
                Q.append(q)
            
            sym = letters[i + 13]
            piece1P.append(sym)
            piece2P.append(sym)
            piece3P = piece3P + sym
            piece4P = piece4P + sym
            sub = ''.join(letters[i + 13] if j == i else letters[j] for j in range(len(t_shape)))
            exprGs.append(f"{sub},{sub}->{sym}")
        else:
            # use triangular matrix as preconditioner for this dim
            q_sharding = None
            if have_qs_sharding:
                if have_hax:
                    # if we're in haliax we can grab fsdp axis and shard accordingly
                    # get current mesh
                    mesh = hax.partitioning._get_mesh()
                    if mesh.devices.shape == ():
                        mesh = None
                    # get fsdp mesh axis
                    if mesh is not None:
                        fsdp_axis_name = hax.partitioning.ResourceAxis.DATA
                        fsdp_axis = mesh.axis_names.index(fsdp_axis_name)
                        fsdp_size = mesh.devices.shape[fsdp_axis]
                        if size % fsdp_size == 0:
                            q_sharding = PartitionSpec(fsdp_axis_name, None)
                        else:
                            q_sharding = PartitionSpec(None, None)
                    else:
                        q_sharding = PartitionSpec(None, None)
                else:
                    # infer a so-so sharding scheme from params if nothing specified
                    # (first dim of q will match corresponding dim in params)
                    q_sharding = precond_sharding if precond_sharding is not None else PartitionSpec(dim_sh, None)
                    # TODO ensure array axis is divisible by mesh axis
                sharding_out[i] = q_sharding

            if existing_Q is None:
                q = scale * jnp.eye(size, dtype=dtype)
                if have_qs_sharding:
                    q = _safe_sharding_constraint(q, q_sharding)
                Q.append(q)

            a = letters[i]
            b = letters[i + 13]
            c = letters[i + 26]
            piece1P.append(a + b)
            piece2P.append(a + c)
            piece3P = piece3P + c
            piece4P = piece4P + b
            sub1 = ''.join(letters[i + 13] if j == i else letters[j] for j in range(len(t_shape)))
            sub2 = ''.join(letters[i + 26] if j == i else letters[j] for j in range(len(t_shape)))
            exprGs.append(f"{sub1},{sub2}->{b}{c}")

    exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return (exprP, exprGs), sharding_out
    return Q, L, (exprP, exprGs), sharding_out


def _norm_lower_bound(A: jax.Array):
    def calc(A):
        # a little faster than xilin's but still pretty accurate
        aa = A * A
        row_norms_sq = jnp.sum(aa, axis=0)
        i = jnp.argmax(row_norms_sq, 0)
        max_row_norm_sq = jax.lax.dynamic_index_in_dim(row_norms_sq, i, 0, keepdims=False)
        max_row_norm = jnp.sqrt(max_row_norm_sq)
        v_r = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False) / max_row_norm
        u_r = A @ v_r
        return jnp.linalg.norm(u_r)

    return jnp.where(jnp.max(jnp.abs(A)) > 0, calc(A), 0.0)


def _update_precond(Q, L, G, exprs, precond_lr, qs_sharding, params_sharding):
    """Update Q using QUAD method and return preconditioned gradient."""
    exprP, exprGs = exprs

    Pg = jnp.einsum(exprP, *Q, *Q, G)
    
    total_numel = G.size
    betaL = 0.95
    def _update_single_q_l(i, q, l):
        term1 = jnp.einsum(exprGs[i], Pg, Pg)
        
        if q.ndim < 2:
            term2 = total_numel / q.size
            ell = jnp.max(jnp.real(term1)) + term2
            l_new = jnp.maximum(betaL * l + (1 - betaL) * ell, ell)
            gain = 1 - precond_lr / 2 / l_new * (term1 - term2)
            q_new = q * gain * gain
        else:
            if qs_sharding is not None:
                sharding = qs_sharding[i]
                if len(sharding) < 2:
                    sharding = PartitionSpec(*((None,) + sharding))
                else:
                    assert len(sharding) == 2
                    sharding = PartitionSpec(*(sharding[1:] + sharding[:1]))
                term1 = _safe_sharding_constraint(term1, sharding)
            
            term2 = total_numel / q.shape[0]  # times I
            ell = _norm_lower_bound(term1) + term2
            l_new = jnp.maximum(betaL * l + (1 - betaL) * ell, ell)
            p = q - precond_lr / 2 / l_new * (term1 @ q - term2 * q)
            p = p - precond_lr / 2 / l_new * (p @ term1 - p * term2)
            q_new = (p + p.T) / 2
            
        return q_new, l_new

    Q_L_new = [_update_single_q_l(i, q, l) for i, (q, l) in enumerate(zip(Q, L))]
    Q_new = [ql[0] for ql in Q_L_new]
    L_new = [ql[1] for ql in Q_L_new]
    
    return Q_new, L_new, Pg


def _safe_sharding_constraint(x, sharding):
    if sharding is None:
        return x
    else:
        return with_sharding_constraint(x, sharding)


def _map_fn(lax_map, bs, n_maps, fn, *args):
    """Maybe map a fn along multiple leading axes."""
    if n_maps <= 0:
        return fn(*args)

    if lax_map:
        mapped_fn = lambda xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return jax.lax.map(mapped_fn, xs=args, batch_size=bs if bs > 1 else None)
    else:
        mapped_fn = lambda *xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return vmap(mapped_fn)(*args)


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
    """

    def __init__(self, param_shape, block_size, dim_diag):
        assert len(dim_diag) == len(param_shape), "dim_diag must have same length as param_shape"
        self._shape = param_shape
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param_shape):
            if 0 < block_size < d and not dim_diag[i]:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._split_sizes = split_sizes

        # TODO (evanatyourservice)
        # this might fail with scalar params but for now we're reshaping those
        single_shape = [a[0] for a in split_sizes]
        padded_single_shape = [-(-dim // block_size) * block_size for dim in single_shape]
        stack_size = max(1, np.prod([max(1, len(s)) for s in split_sizes]))
        self._padded_stacked_shape = tuple([stack_size] + padded_single_shape)

    def split_sizes(self):
        return self._split_sizes

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for i, indices in self._splits:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
            tensors = tensors_local
        return tuple(tensors)

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for i, indices in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(jnp.concatenate(partitions[ind : ind + n], axis=i))
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


def _partitions(lst):
    """Generate all partitions of a list."""
    if not lst:
        yield [[]]
    else:
        for i in range(len(lst)):
            for part in _partitions(lst[i + 1 :]):
                yield [lst[: i + 1]] + part


def _merge_small_dims(
    shape_to_merge, max_dim, dim_diag, sharding_to_merge=None
) -> Tuple[List[int], List[bool], Optional[PartitionSpec]]:
    if not shape_to_merge:  # handles scalar shape ()
        return [], [True], PartitionSpec() if sharding_to_merge is not None else None
    if np.all(np.array(shape_to_merge) == 1):  # handles shape (1,)
        return (
            [1],
            [True],
            PartitionSpec(None) if sharding_to_merge is not None else None,
        )

    def dim2loss(d, dim0=max_dim):
        """A heuristic map from dim to loss with the least loss occurs at dim0."""
        loss = 0
        if d < dim0:
            loss += np.log2(dim0 / d)
            too_small = dim0 / 8
            if d < too_small:
                loss += 100 * np.log2(too_small / d)
        else:
            loss += 10 * np.log2(d / dim0)
            too_large = 8 * dim0
            if d > too_large:
                loss += 1000 * np.log2(d / too_large)
        return loss

    best_loss = float("inf")
    best_partition = []

    for p in _partitions(list(range(len(shape_to_merge)))):
        loss = 0
        merged = []
        for group in p:
            if not group:
                continue
            d = np.prod([shape_to_merge[i] for i in group])
            loss += dim2loss(d)
            merged.append(group)
        # skip partitions that would merge everything into a single dimension
        # when we started with 2 or more dimensions
        if len(shape_to_merge) >= 2 and len(merged) == 1:
            continue
        if loss < best_loss:
            best_loss = loss
            best_partition = merged

    merged_shape = []
    merged_diag = []
    merged_sharding: List[Union[tuple, None]] = []

    for group in best_partition:
        merged_shape.append(np.prod([shape_to_merge[i] for i in group]))
        merged_diag.append(all(dim_diag[i] for i in group))
        if sharding_to_merge:
            group_shardings = [sharding_to_merge[i] for i in group]
            valid_shardings = [s for s in group_shardings if s is not None]

            if len(valid_shardings) > 1:
                merged_sharding.append(tuple(valid_shardings))
            elif len(valid_shardings) == 1:
                merged_sharding.append(valid_shardings[0])
            else:
                merged_sharding.append(None)

    return (
        merged_shape,
        merged_diag,
        PartitionSpec(*merged_sharding) if sharding_to_merge else None,
    )


def _pad_and_stack_matrices(array_list, block_size):
    # Handle scalar arrays by adding a dummy dimension
    is_scalar = len(array_list[0].shape) == 0
    if is_scalar:
        array_list = [arr[None] for arr in array_list]

    shapes = [arr.shape for arr in array_list]
    max_dims = [max(shape[i] for shape in shapes) for i in range(len(shapes[0]))]
    padded_shape = [-(-dim // block_size) * block_size for dim in max_dims]
    padded_arrays = []
    for arr in array_list:
        pad_width = [(0, padded_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
        padded = jnp.pad(arr, pad_width)
        padded_arrays.append(padded)

    stacked = jnp.stack(padded_arrays)
    return stacked


def _unstack_and_unpad_matrices(stacked_array, original_shapes):
    # Handle scalar arrays
    is_scalar = len(original_shapes[0]) == 0

    unstacked = jnp.split(stacked_array, stacked_array.shape[0], axis=0)
    unpadded = []
    for arr, orig_shape in zip(unstacked, original_shapes):
        arr = jnp.squeeze(arr, axis=0)
        if is_scalar:
            # For scalars, just take the first element
            arr = arr[0]
        else:
            # For non-scalars, slice to original shape
            slices = tuple(slice(0, dim) for dim in orig_shape)
            arr = arr[slices]
        unpadded.append(arr)
    return tuple(unpadded)


# unused fns (can be used for stacking partitions without padding):
def _sort_and_group_matrices(matrix_shapes: List[Tuple[int, ...]]):
    indexed_list = list(enumerate(matrix_shapes))
    sorted_indexed = sorted(indexed_list, key=lambda x: x[1])
    sorted_shapes = [shape for _, shape in sorted_indexed]
    change_indices = [original_index for original_index, _ in sorted_indexed]
    revert_indices = [0] * len(matrix_shapes)
    for new_pos, (original_index, _) in enumerate(sorted_indexed):
        revert_indices[original_index] = new_pos
    shape_groups = defaultdict(list)
    for i, shape in enumerate(sorted_shapes):
        shape_groups[shape].append(i)
    unique_sorted_shapes = list(shape_groups.keys())
    return unique_sorted_shapes, dict(shape_groups), change_indices, revert_indices


def _stack_matrices(array_list):
    in_tuple = isinstance(array_list, tuple)
    shapes = [arr.shape for arr in array_list]
    unique_shapes, shape_groups, change_indices, _ = _sort_and_group_matrices(shapes)
    sorted_arrays = [array_list[i] for i in change_indices]
    stacked_arrays = []
    for shape in unique_shapes:
        indices = shape_groups[shape]
        stacked = jnp.stack([sorted_arrays[i] for i in indices])
        stacked_arrays.append(stacked)
    if in_tuple:
        return tuple(stacked_arrays)
    return stacked_arrays


def _unstack_matrices(stacked_arrays, revert_indices):
    in_tuple = isinstance(stacked_arrays, tuple)
    unstacked = []
    for arr in stacked_arrays:
        unstacked.extend(jnp.split(arr, arr.shape[0]))
    array_list = [jnp.squeeze(unstacked[i], axis=0) for i in revert_indices]
    if in_tuple:
        return tuple(array_list)
    return array_list
