import abc
import functools
import inspect
import typing
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, NamedTuple, Optional, TypeVar, Union, runtime_checkable

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

import levanter.tracker
from levanter.utils.jax_utils import parameter_count


T = TypeVar("T")
M = TypeVar("M")
Ex = TypeVar("Ex")


@runtime_checkable
class SophiaGObjective(typing.Protocol):
    """
    Class for objective functions that can be used with Sophia-G

    Sophia-G is a second order optimizer that uses the Gauss-Newton-Bartlett approximation to the Hessian
    to compute the second order update. This requires the objective function be of the form loss(logits(x))
    where logits(x) is the activation of the model for the given example x. This is the case for most models
    that are trained with "typical" losses.
    """

    def logits(self, parameters: M, example: Ex, *args, **kwargs) -> Any:
        """
        Returns the logits/activations of the model for the given example,
        or just sufficient statistics for the example for non-categorical models.
        """
        ...

    def sample(self, logits, example: Ex, *, key: PRNGKey) -> Ex:
        """
        Samples a new example with the same shape as the original example, but with
        the "labels" replaced with some sampled values
        """
        ...

    def loss(self, logits, example: Ex):
        """
        Just computes the loss, e.g. cross entropy.

        Should return the mean loss over the batch, not the sum.

        TODO: should we reconsider this?
        """
        ...

    def __call__(self, parameters: M, example: Ex, *args, **kwargs):
        """
        Just a convenience method for invoking the objective for "normal" training w/o sophia-g
        """
        logits = self.logits(parameters, example, *args, **kwargs)
        return self.loss(logits, example)

    def num_data_points(self, example: Ex) -> int:
        """
        Returns the number of data points in the example. This should take into account the loss mask
        or any other masking that might be applied to the example.

        By default, we just return 1, and you can just pull the term into the hyperparams of Sophia if you want.

        Returns:
               The number of data points in the example
        """
        return 1


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    learning_rate: float = 6e-4
    weight_decay: float = 0.0

    min_lr_ratio: float = 0.1
    warmup_ratio: Optional[float] = None  # Deprecated. fraction of training steps to use as warmup
    warmup: float = 0.01
    """fraction of training steps to use as warmup, or steps to use. 0.0 means no warmup"""
    cooldown: float = 0.0
    """fraction of training steps to use as cooldown, or steps to use. 0.0 means no cooldown"""
    lr_schedule: str = "cosine"  # constant, cosine, linear

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "adam"

    @abc.abstractmethod
    def build(self, num_train_steps: int):
        raise NotImplementedError

    def lr_scheduler(self, num_train_steps):
        warmup_steps = self._convert_warmup(num_train_steps)
        cooldown_steps = _convert_ratio_or_steps(self.cooldown, num_train_steps)
        lr_decay_steps = num_train_steps - warmup_steps - cooldown_steps
        min_lr = self.learning_rate * self.min_lr_ratio

        match self.lr_schedule:
            case "constant":
                schedule = optax.constant_schedule(self.learning_rate)
            case "cosine":
                schedule = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
            case "linear":
                schedule = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps - warmup_steps)
            case "inv_sqrt":
                schedule = _inv_sqrt_decay_schedule(self.learning_rate, min_lr, warmup_steps, 10000)
            case _:
                raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

        schedules = []
        boundaries = []

        if warmup_steps != 0:
            warmup = optax.linear_schedule(0.0, self.learning_rate, warmup_steps)
            schedules.append(warmup)
            boundaries.append(warmup_steps)

        schedules.append(schedule)

        if cooldown_steps != 0:
            final_main_lr = schedule(lr_decay_steps)
            cooldown = optax.linear_schedule(final_main_lr, min_lr, cooldown_steps)
            schedules.append(cooldown)
            boundaries.append(num_train_steps - cooldown_steps)

        if len(schedules) > 1:
            schedule = optax.join_schedules(schedules, boundaries)

        return schedule

    def _convert_warmup(self, num_train_steps: int):
        if self.warmup_ratio is not None:
            warnings.warn("warmup_ratio is deprecated. Use warmup instead")
            return int(self.warmup_ratio * num_train_steps)
        else:
            return _convert_ratio_or_steps(self.warmup, num_train_steps)


def _inv_sqrt_decay_schedule(lr: float, min_lr: float, warmup_steps: int, timescale: float = 10000):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.maximum(count + warmup_steps, 1) / timescale))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _convert_ratio_or_steps(ratio_or_steps: float, num_train_steps: int):
    if ratio_or_steps < 1.0:
        return int(ratio_or_steps * num_train_steps)
    else:
        return int(ratio_or_steps)


@dataclass
class HessianOptConfig(OptimizerConfig, abc.ABC):
    update_interval: int = 10
    """How often to update the hessian approximation."""


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    weight_decay: float = 0.1
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


GAMMA_SOPHIA_G = 0.05
GAMMA_SOPHIA_H = 0.01


@dataclass
class BaseSophiaConfig(HessianOptConfig):
    """Base class for sophia variants. Doesn't implement the state update"""

    weight_decay: float = 0.1
    beta1: float = 0.96
    beta2: float = 0.99

    epsilon: float = 1e-12
    clip_threshold: Optional[float] = 1.0

    @abc.abstractmethod
    def compute_hessian(
        self,
        fn,
        model,
        *batch,
        hess_key: PRNGKey,
        **batch_kwargs,
    ):
        raise NotImplementedError

    def build(self, num_train_steps: int):
        def _optimizer(learning_rate, gamma) -> SecondOrderTransformation:
            components = []

            components.append(
                _sophia_gradient_transform(
                    sophia_hess_fn=self.compute_hessian,
                    update_interval=self.update_interval,
                    b1=self.beta1,
                    b2=self.beta2,
                    eps=self.epsilon,
                    gamma=gamma,
                    clip_threshold=self.clip_threshold,
                )
            )

            # Algorithm 3, step 11 (Note, this comes after clipping b/c it's not supposed to be clipped)
            # In the paper, it comes as a prior step, but doesn't get clipped
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
class SophiaGConfig(BaseSophiaConfig):
    gamma: float = GAMMA_SOPHIA_G

    def compute_hessian(self, fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
        return stochastic_diag_gauss_newton(fn, model, *batch, **batch_kwargs, hess_key=hess_key)


@OptimizerConfig.register_subclass("sophia-h")
@dataclass
class SophiaHConfig(BaseSophiaConfig):
    gamma: float = GAMMA_SOPHIA_H

    def compute_hessian(self, fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
        return stochastic_hessian_diagonal(fn, model, *batch, **batch_kwargs, hess_key=hess_key)


class ScaleByHessianState(NamedTuple):
    """State for Sophia and similar."""

    count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    hessian_count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    mu: optax.Updates  # momentum
    h: optax.Updates  # EMA of hessian diagonal


class HessianUpdateFn(typing.Protocol):
    """A callable type for the"""

    def __call__(
        self,
        state,
        fn,
        model,
        *batch,
        hess_key: PRNGKey,
        **batch_kwargs,
    ) -> optax.OptState:
        """Returns the updated `state` given the `hessian` and `state`."""
        pass


class SecondOrderTransformation(NamedTuple):
    """A triple of pure functions that together define a second-order optimizer."""

    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    hessian_update: HessianUpdateFn


AnySecondOrderTransformation = Union[SecondOrderTransformation, optax.GradientTransformation]


# TODO: filter_jvp?
def hvp(f, x, v):
    """Compute the Hessian-vector product of a function."""
    return jax.jvp(eqx.filter_grad(f), (x,), (v,))[1]


# Use this for Sophia-H
def stochastic_hessian_diagonal(fn, model, *args, hess_key: PRNGKey, **kwargs):
    """Compute the diagonal of the Hessian of a function using a normal distribution.

    https://arxiv.org/pdf/2305.14342.pdf Algorithm 1

    Args:
        fn: function to compute the Hessian of
        model: model to compute the Hessian of
        hess_key: key for the normal distribution
    """
    # cf https://arxiv.org/pdf/2006.00719.pdf eqn 9
    # https://www-users.cse.umn.edu/~saad/PDF/umsi-2005-082.pdf
    # https://arxiv.org/pdf/2208.03268.pdf
    g = tree_gaussian(hess_key, model)
    # TODO: consider allowing for n > 1 gaussians?
    product = hvp(lambda m: fn(m, *args, **kwargs), model, g)
    hessian = jax.tree_util.tree_map(lambda grad, gaussian: grad * gaussian, product, g)

    return hessian


# use this for Sophia-G
def stochastic_diag_gauss_newton(fn: SophiaGObjective, model, example, *args, hess_key: PRNGKey, **kwargs):
    """

    Approximate the diagonal of the Hessian using an approximation to the Gauss Newton matrix.
    This is Algorithm 2 of https://arxiv.org/pdf/2305.14342.pdf

    Args:
        fn (SophiaGObjective): objective function
        model: model whose Hessian to compute
        hess_key: key for sampling
        *args, **kwargs: passed to fn's logits
    """
    if not isinstance(fn, SophiaGObjective):
        raise ValueError("objective must be a SophiaGObjective")

    # Step 3
    logits, model_backward = eqx.filter_vjp(lambda model: fn.logits(model, example, *args, **kwargs), model)

    # Step 4
    y_hat = fn.sample(logits, example, key=hess_key)

    # Step 5
    grad_loss_logits = eqx.filter_grad(fn.loss)(logits, y_hat)
    pseudo_g = model_backward(grad_loss_logits)[0]

    # Step 6
    bs = fn.num_data_points(example)
    h = jax.tree_util.tree_map(lambda x: x**2 * bs, pseudo_g)

    return h


def tree_gaussian(key, tree):
    """Samples a tree of gaussian noise with the same structure as `tree`."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    g = jax.tree_util.tree_map(lambda x, key: jax.random.normal(key, x.shape), leaves, list(keys))
    g = jax.tree_util.tree_unflatten(structure, g)

    return g


def sophia_h(
    lr: float = 0.85e-3,
    *,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = GAMMA_SOPHIA_H,
    weight_decay: float = 0.0,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
) -> SecondOrderTransformation:
    """Sophia-H: https://arxiv.org/pdf/2305.14342.pdf Algorithm 1&3"""
    components = []

    components.append(scale_by_sophia_h(b1, b2, eps, gamma, clip_threshold, update_interval))

    if weight_decay > 0:
        components.append(optax.add_decayed_weights(weight_decay))

    components.append(optax.scale(-lr))

    return chain_second_order(*components)


def scale_by_sophia_h(
    b1=0.965, b2=0.99, eps=1e-8, gamma=GAMMA_SOPHIA_H, clip_threshold: Optional[float] = 1.0, update_interval=10
):

    return _sophia_gradient_transform(
        sophia_hess_fn=stochastic_hessian_diagonal,
        update_interval=update_interval,
        b1=b1,
        b2=b2,
        eps=eps,
        gamma=gamma,
        clip_threshold=clip_threshold,
    )


def sophia_g(
    lr: float = 1e-3,
    *,
    b1: float = 0.99,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = GAMMA_SOPHIA_G,
    weight_decay: float = 0.0,
    clip_threshold: Optional[float] = 1.0,
    update_interval: int = 10,
) -> SecondOrderTransformation:
    """Sophia-G: https://arxiv.org/pdf/2305.14342.pdf Algorithm 2&3"""
    components = []

    components.append(scale_by_sophia_g(b1, b2, eps, gamma, clip_threshold, update_interval))

    if weight_decay > 0:
        components.append(optax.add_decayed_weights(weight_decay))

    components.append(optax.scale(-lr))

    return chain_second_order(*components)


def scale_by_sophia_g(
    b1: float = 0.99,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = GAMMA_SOPHIA_G,
    clip_threshold: Optional[float] = 1.0,
    update_interval=10,
):

    return _sophia_gradient_transform(
        sophia_hess_fn=stochastic_diag_gauss_newton,
        update_interval=update_interval,
        b1=b1,
        b2=b2,
        eps=eps,
        gamma=gamma,
        clip_threshold=clip_threshold,
    )


def _sophia_gradient_transform(
    sophia_hess_fn,
    update_interval: int,
    b1: float,
    b2: float,
    eps: float,
    gamma: float,
    clip_threshold: Optional[float],
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
        # track how often hessian is used
        mu_leaves = jax.tree_util.tree_leaves(mu_hat)
        h_leaves = jax.tree_util.tree_leaves(h_hat)

        stats: dict[str, Any] = {
            "optim/param_norm": jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))),
            "optim/momentum_norm": jnp.sqrt(sum(jnp.sum(m**2) for m in mu_leaves)),
            "optim/hessian_norm": jnp.sqrt(sum(jnp.sum(h**2) for h in h_leaves)),
        }

        # with sophia-g the max(h, 0) is not needed but no harm
        updates = jax.tree_util.tree_map(
            # lambda m, v: m / jnp.maximum(jnp.maximum(jnp.abs(m), gamma * jnp.maximum(v, 0)), eps), mu_hat, h_hat
            lambda m, h: m / jnp.maximum(gamma * h, eps),
            mu_hat,
            h_hat,
        )

        if clip_threshold is not None:
            unclipped_count = sum(jnp.sum(jnp.abs(u) < clip_threshold) for u in jax.tree_util.tree_leaves(updates))
            updates = jax.tree_util.tree_map(lambda u: jnp.clip(u, -clip_threshold, clip_threshold), updates)
            stats["optim/unclipped_fraction"] = unclipped_count / parameter_count(updates)

        # this doesn't work well on CPU, so skip if cpu
        if jax.lib.xla_bridge.get_backend().platform != "cpu":
            levanter.tracker.jit_log_metrics(stats, step=state.count)

        if mu_dtype is not None:
            mu = jax.tree_util.tree_map(lambda t: t.astype(mu_dtype), mu)

        return updates, ScaleByHessianState(count=count_inc, hessian_count=state.hessian_count, mu=mu, h=h_hat)

    def update_hessian(state, fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
        def _do_update():
            new_hess = sophia_hess_fn(fn, model, *batch, hess_key=hess_key, **batch_kwargs)
            new_hess = jax.tree_util.tree_map(lambda h: jnp.clip(h, -1, 1), new_hess)

            # EMAs of hessian
            hessian_count_inc = numerics.safe_int32_increment(state.hessian_count)
            nu = update_moment(new_hess, state.h, b2, 1)
            return ScaleByHessianState(count=state.count, hessian_count=hessian_count_inc, mu=state.mu, h=nu)

        def _dont_update():
            return state

        return jax.lax.cond(
            jnp.equal(state.count % update_interval, 0),
            lambda _: _do_update(),
            lambda _: _dont_update(),
            state.count,
        )

    return SecondOrderTransformation(init_fn, update_fn, update_hessian)


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

    def hessian_update_fn(state, fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
        if len(hessian_update_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in chain! Make sure you have called init first!"
            )

        new_state = []
        for s, update_fn in zip(state, hessian_update_fns):
            if update_fn is None:
                new_state.append(s)
            else:
                new_s = update_fn(s, fn, model, *batch, hess_key=hess_key, **batch_kwargs)
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

        def update_hessian(state, fn, model, *batch, hess_key: PRNGKey, **batch_kwargs):
            if hyperparam_dtype is None:
                dtype = getattr(next(iter(jax.tree_util.tree_leaves(state)), None), "dtype", None)
            else:
                dtype = hyperparam_dtype
            hparams = {k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(state.count, dtype))
            new_inner_state = inner_factory(**other_hps, **hparams).hessian_update(
                state.inner_state,
                fn,
                model,
                *batch,
                hess_key=hess_key,
                **batch_kwargs,
            )

            # pylint:disable=too-many-function-args
            return InjectHyperparamsState(state.count, hparams, new_inner_state)
            # pylint:enable=too-many-function-args

        return SecondOrderTransformation(init_fn, update_fn, update_hessian)

    return wrapped_transform
