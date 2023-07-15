# implements the prototype sofia optimizer
import abc
import functools
import inspect
import typing
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, Union

import chex
import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
import optax
from jax.random import PRNGKey
from optax._src import numerics
from optax._src.schedule import InjectHyperparamsState, _convert_floats
from optax._src.transform import bias_correction, update_moment

import haliax

from levanter.logging import jittable_wandb_log
from levanter.utils.jax_utils import parameter_count


class SophiaGLossFn(typing.Protocol):
    def __call__(self, logits: jaxtyping.PyTree, labels: jaxtyping.PyTree) -> jnp.ndarray:
        ...


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    learning_rate: float = 6e-4

    min_lr_ratio: float = 0.0
    warmup_ratio: float = 0.01  # fraction of training steps to use as warmup
    lr_schedule: str = "cosine"  # constant, cosine, linear

    @abc.abstractmethod
    def build(self, num_train_steps: int):
        raise NotImplementedError

    def lr_scheduler(self, num_train_steps):
        warmup_steps = int(self.warmup_ratio * num_train_steps)
        lr_decay_steps = num_train_steps - warmup_steps
        min_lr = self.learning_rate * self.min_lr_ratio

        if self.lr_schedule == "constant":
            schedule = optax.constant_schedule(self.learning_rate)
        elif self.lr_schedule == "cosine":
            schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
        elif self.lr_schedule == "linear":
            schedule = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps - warmup_steps)
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        if warmup_steps != 0:
            warmup = optax.linear_schedule(0.0, self.learning_rate, warmup_steps)
            schedule = optax.join_schedules([warmup, schedule], [warmup_steps])

        return schedule


@dataclass
class HessianOptConfig(OptimizerConfig, abc.ABC):
    state_update_interval: int = 10

    @abc.abstractmethod
    def opt_state_update(
        self, optimizer, opt_state, loss_fn: SophiaGLossFn, model, *batch, hess_key: PRNGKey, **batch_kwargs
    ):
        raise NotImplementedError


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                # TODO: add weight decay masking??
                components.append(optax.add_decayed_weights(self.weight_decay))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


GAMMA_SOPHIA_G = 2e4
GAMMA_SOPHIA_H = 0.1


@dataclass
class BaseSophiaConfig(HessianOptConfig):
    """Base class for sophia variants. Doesn't implement the state update"""

    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999

    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0

    def build(self, num_train_steps: int):
        def _optimizer(learning_rate, gamma) -> SecondOrderTransformation:
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(scale_by_sophia(b1=self.beta1, b2=self.beta2, eps=self.epsilon, gamma=gamma))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = chain_second_order(*components)

            return optimizer

        # Hong suggested using cosine decay for gamma
        gamma_decay_schedule = optax.cosine_decay_schedule(self.gamma, num_train_steps // 2, 0)  # type: ignore
        constant_gamma_schedule = optax.constant_schedule(self.gamma)  # type: ignore
        gamma_schedule = optax.join_schedules([constant_gamma_schedule, gamma_decay_schedule], [num_train_steps // 2])

        return inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps), gamma=gamma_schedule)


@OptimizerConfig.register_subclass("sophia-g")
@dataclass
class SophiaGConfig(HessianOptConfig):
    gamma: float = GAMMA_SOPHIA_G
    logit_axis: str = "vocab"  # axis to sample from, from logits

    def opt_state_update(self, optimizer, opt_state, loss_fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
        hess = stochastic_diag_gauss_newton(
            loss_fn, model, *batch, **batch_kwargs, g_key=hess_key, Vocab=self.logit_axis
        )

        # distribute gradients across the mesh and apply them
        opt_state = optimizer.hessian_update(hess, opt_state)

        return opt_state


@OptimizerConfig.register_subclass("sophia-h")
@dataclass
class SophiaHConfig(HessianOptConfig):
    gamma: float = GAMMA_SOPHIA_H

    def opt_state_update(self, optimizer, opt_state, loss_fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
        hess = stochastic_hessian_diagonal(loss_fn, model, *batch, **batch_kwargs, h_key=hess_key)

        # distribute gradients across the mesh and apply them
        opt_state = optimizer.hessian_update(hess, opt_state)

        return opt_state


class ScaleByHessianState(NamedTuple):
    """State for Sophia and similar."""

    count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    hessian_count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates  # momentum
    h: optax.Updates  # EMA of hessian diagonal


class HessianUpdateFn(typing.Protocol):
    """A callable type for the `update` step of a `GradientTransformation`.

    The `update` step takes a tree of candidate parameter `updates` (e.g. their
    gradient with respect to some loss), an arbitrary structured `state`, and the
    current `params` of the model being optimised. The `params` argument is
    optional, it must however be provided when using transformations that require
    access to the current values of the parameters.
    """

    def __call__(
        self,
        hessian: optax.Updates,
        state: optax.OptState,
    ) -> optax.OptState:
        """Returns the updated `state` given the `hessian` and `state`."""
        pass


class SecondOrderTransformation(NamedTuple):
    """A pair of pure functions implementing a second order gradient transformation."""

    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    hessian_update: HessianUpdateFn


AnySecondOrderTransformation = Union[SecondOrderTransformation, optax.GradientTransformation]


# TODO: filter_jvp?
def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return jax.jvp(eqx.filter_grad(f), (x,), (v,))[1]


# Use this for Sophia-H
def stochastic_hessian_diagonal(fn, model, *args, g_key: PRNGKey, **kwargs):
    """Compute the diagonal of the Hessian of a function using a normal distribution.

    Args:
        fn: function to compute the Hessian of
        model: model to compute the Hessian of
        g_key: key for the normal distribution
    """
    # cf https://arxiv.org/pdf/2006.00719.pdf eqn 9
    # https://www-users.cse.umn.edu/~saad/PDF/umsi-2005-082.pdf
    # https://arxiv.org/pdf/2208.03268.pdf
    g = tree_gaussian(g_key, model)
    # TODO: consider allowing for n > 1 gaussians
    product = hvp(lambda m: fn(m, *args, **kwargs), model, g)
    hessian = jax.tree_util.tree_map(lambda grad, gaussian: grad * gaussian, product, g)

    return hessian


# use this for Sophia-G
def stochastic_diag_gauss_newton(fn, model, *args, g_key: PRNGKey, Vocab: haliax.AxisSelector, **kwargs):
    """Approximate the diagonal of the Hessian using an approximation to the Gauss Newton matrix.

    Args:
        fn: loss function of the form fn(logits, y). Should return a scalar, even if batched
        model:
        g_key: key for sample from
    """
    logits, model_backward = eqx.filter_vjp(lambda f: f(*args, **kwargs), model)
    hat_y = haliax.random.categorical(g_key, logits, Vocab)

    grad_loss_logits = eqx.filter_grad(fn)(logits, hat_y)
    pseudo_g = model_backward(grad_loss_logits)[0]

    return jax.tree_util.tree_map(lambda x: x**2, pseudo_g)


def tree_gaussian(key, tree):
    """Samples a tree of gaussian noise with the same structure as `tree`."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    g = jax.tree_util.tree_map(lambda x, key: jax.random.normal(key, x.shape), leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def scale_by_sophia(
    b1: float = 0.96,
    b2: float = 0.99,
    eps: float = 1e-12,
    gamma: float = 0.01,
    mu_dtype: Optional[Any] = None,
) -> SecondOrderTransformation:
    mu_dtype = jax.canonicalize_dtype(mu_dtype) if mu_dtype is not None else None

    def init_fn(params):
        mu = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)  # First moment
        h = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        return ScaleByHessianState(count=jnp.zeros([], jnp.int32), hessian_count=jnp.zeros([], jnp.int32), mu=mu, h=h)

    def update_fn(updates, state, params=None):
        mu = update_moment(updates, state.mu, b1, 1)
        # nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        h_hat = state.h
        # TODO: Use slightly lower learning rate for adam, e.g. 0.85 * adam_lr
        # TODO: monitor param norm and momentum norm and trace(hessian) (aka sum of h_hat)
        # TODO: also track how often hessian is used (per coordinate)
        # TODO: track sum( jnp.abs(m) < gamma * jnp.maximum(v, 0) for m in mu_hat), we expect this to be ~70% later in training
        # track how often hessian is used
        mu_leaves = jax.tree_util.tree_leaves(mu_hat)
        h_leaves = jax.tree_util.tree_leaves(h_hat)

        # with sofia+ the max(h, 0) is not needed but no harm
        hessian_use_count = sum(
            jnp.sum(jnp.abs(mu) < gamma * jnp.maximum(h, 0)) for (mu, h) in zip(mu_leaves, h_leaves)
        )

        param_count = parameter_count(updates)
        hessian_use_ratio = hessian_use_count / param_count

        param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params)))
        momentum_norm = jnp.sqrt(sum(jnp.sum(m**2) for m in mu_leaves))
        hessian_norm = jnp.sqrt(sum(jnp.sum(h**2) for h in h_leaves))

        # this doesn't work well on CPU, so skip if cpu
        if jax.lib.xla_bridge.get_backend().platform != "cpu":
            jittable_wandb_log(
                {
                    "optim/hessian_use_ratio": hessian_use_ratio,
                    "optim/param_norm": param_norm,
                    "optim/momentum_norm": momentum_norm,
                    "optim/hessian_norm": hessian_norm,
                },
                step=state.count,
            )

        updates = jax.tree_util.tree_map(
            lambda m, v: m / jnp.maximum(jnp.maximum(jnp.abs(m), gamma * jnp.maximum(v, 0)), eps), mu_hat, h_hat
        )
        if mu_dtype is not None:
            mu = jax.tree_util.tree_map(lambda t: t.astype(mu_dtype), mu)
        return updates, ScaleByHessianState(count=count_inc, hessian_count=state.hessian_count, mu=mu, h=h_hat)

    def update_hessian(hessian, state, params=None):
        del params
        hessian_count_inc = numerics.safe_int32_increment(state.hessian_count)
        nu = update_moment(hessian, state.h, b2, 1)
        return ScaleByHessianState(count=state.count, hessian_count=hessian_count_inc, mu=state.mu, h=nu)

    return SecondOrderTransformation(init_fn, update_fn, update_hessian)


def sophia(
    lr: float,
    b1: float = 0.95,
    b2: float = 0.99,
    eps: float = 1e-12,
    gamma: float = 0.1,
    mu_dtype: Optional[Any] = None,
) -> SecondOrderTransformation:
    return chain_second_order(scale_by_sophia(b1, b2, eps, gamma, mu_dtype), optax.scale(lr))


# what follows are "second order" versions of optax functions
def chain_second_order(*args: AnySecondOrderTransformation) -> SecondOrderTransformation:
    """Applies a list of chainable update transformations. Analogous to optax.chain,
    but for second order transformations.
    """

    init_fns = []
    update_fns = []
    hessian_update_fns: List[Optional[HessianUpdateFn]] = []

    for arg in args:
        if isinstance(arg, SecondOrderTransformation):
            init_fns.append(arg.init)
            update_fns.append(arg.update)
            hessian_update_fns.append(arg.hessian_update)
        else:
            init_fns.append(arg.init)
            update_fns.append(arg.update)
            hessian_update_fns.append(None)

    def init_fn(params):
        return tuple(fn(params) for fn in init_fns)

    def update_fn(updates, state, params=None):
        if len(update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in chain! Make sure you have called init first!"
            )

        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params)
            new_state.append(new_s)
        return updates, tuple(new_state)

    def hessian_update_fn(new_hessian, state, params=None):
        if len(hessian_update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in chain! Make sure you have called init first!"
            )

        new_state = []
        for s, fn in zip(state, hessian_update_fns):
            if fn is None:
                new_state.append(s)
            else:
                new_s = fn(new_hessian, s, params)
                new_state.append(new_s)
        return tuple(new_state)

    return SecondOrderTransformation(init_fn, update_fn, hessian_update_fn)


def inject_hyperparams(
    inner_factory: Callable[..., SecondOrderTransformation],
    static_args: Union[str, Iterable[str]] = (),
    hyperparam_dtype: Optional[jnp.dtype] = None,
) -> Callable[..., SecondOrderTransformation]:
    """
    Second Order version of optax.inject_hyperparams.

    Original docstring:

    Wrapper that injects hyperparameters into the inner GradientTransformation.

    This wrapper allows you to pass schedules (i.e. a function that returns a
    numeric value given a step count) instead of constants for
    hyperparameters. You may only schedule numeric hyperparameters (i.e. boolean
    flags cannot be scheduled).

    For example, to use ``scale_by_adam`` with a piecewise linear
    schedule for beta_1 and constant for beta_2::

      scheduled_adam = optax.inject_hyperparams(optax.scale_by_adam)(
          b1=optax.piecewise_linear_schedule(...),
          b2=0.99)

    You may manually change numeric hyperparameters that were not scheduled
    through the ``hyperparams`` dict in the ``InjectHyperparamState``::

      state = scheduled_adam.init(params)
      updates, state = scheduled_adam.update(grads, state)
      state.hyperparams['b2'] = 0.95
      updates, state = scheduled_adam.update(updates, state)  # uses b2 = 0.95

    Manually overriding scheduled hyperparameters will have no effect (e.g.
    in the code sample above, you cannot manually adjust ``b1``).

    Args:
      inner_factory: a function that returns the inner
        ``optax.GradientTransformation`` given the hyperparameters.
      static_args: a string or iterable of strings specifying which
        callable parameters are not schedules. inject_hyperparams treats all
        callables as schedules by default, so if a hyperparameter is a
        non-schedule callable, you must specify that using this argument.
      hyperparam_dtype: Optional datatype override. If specified, all float
        hyperparameters will be cast to this type.

    Returns:
      A callable that returns a ``optax.GradientTransformation``. This callable
      accepts the same arguments as ``inner_factory``, except you may provide
      schedules in place of the constant arguments.
    """
    static_args = {static_args} if isinstance(static_args, str) else set(static_args)
    inner_signature = inspect.signature(inner_factory)

    if not static_args.issubset(inner_signature.parameters):
        raise ValueError(
            "`static_args` must specify a subset of `inner_factory`'s parameters. "
            f"Given `static_args`: {static_args}. `inner_factory` parameters: "
            f"{set(inner_signature.parameters.keys())}"
        )

    @functools.wraps(inner_factory)
    def wrapped_transform(*args, **kwargs) -> SecondOrderTransformation:
        bound_arguments = inner_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        sched_hps, numeric_hps, other_hps = {}, {}, {}
        for name, value in bound_arguments.arguments.items():
            if name in static_args or isinstance(value, bool):
                other_hps[name] = value
            elif callable(value):
                sched_hps[name] = value
            elif isinstance(value, (int, float, chex.Array)):
                numeric_hps[name] = value
            else:
                other_hps[name] = value

        def schedule_fn(count, dtype):
            return {k: _convert_floats(f(count), dtype) for k, f in sched_hps.items()}

        def init_fn(params):
            count = jnp.zeros([], jnp.int32)
            if hyperparam_dtype is None:
                dtype = getattr(next(iter(jax.tree_util.tree_leaves(params)), None), "dtype", None)
            else:
                dtype = hyperparam_dtype
            hparams = {k: jnp.asarray(_convert_floats(v, dtype)) for k, v in numeric_hps.items()}
            hparams.update(schedule_fn(count, dtype))
            return InjectHyperparamsState(  # pylint:disable=too-many-function-args
                count, hparams, inner_factory(**other_hps, **hparams).init(params)
            )

        def update_fn(updates, state, params=None):
            if hyperparam_dtype is None:
                dtype = getattr(next(iter(jax.tree_util.tree_leaves(updates)), None), "dtype", None)
            else:
                dtype = hyperparam_dtype
            hparams = {k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(state.count, dtype))
            updates, inner_state = inner_factory(**other_hps, **hparams).update(updates, state.inner_state, params)
            count_inc = numerics.safe_int32_increment(state.count)

            # pylint:disable=too-many-function-args
            return updates, InjectHyperparamsState(count_inc, hparams, inner_state)
            # pylint:enable=too-many-function-args

        def update_hessian(hessian, state):
            if hyperparam_dtype is None:
                dtype = getattr(next(iter(jax.tree_util.tree_leaves(hessian)), None), "dtype", None)
            else:
                dtype = hyperparam_dtype
            hparams = {k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(state.count, dtype))
            new_inner_state = inner_factory(**other_hps, **hparams).hessian_update(hessian, state.inner_state)

            # pylint:disable=too-many-function-args
            return InjectHyperparamsState(state.count, hparams, new_inner_state)
            # pylint:enable=too-many-function-args

        return SecondOrderTransformation(init_fn, update_fn, update_hessian)

    return wrapped_transform
