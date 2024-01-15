import functools
import inspect
import typing
from typing import Callable, Iterable, List, NamedTuple, Optional, Union

import chex
import jax
import optax
from jax import numpy as jnp
from optax import InjectHyperparamsState


class HessianUpdateFn(typing.Protocol):
    """A callable type for the"""

    def __call__(
        self,
        state,
        fn,
        model,
        *batch,
        **batch_kwargs,
    ) -> optax.OptState:
        """Returns the updated `state` given the `hessian` and `state`."""
        pass


class SecondOrderTransformation(NamedTuple):
    """A triple of pure functions that together define a second-order optimizer."""

    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    update_hessian: HessianUpdateFn


AnySecondOrderTransformation = Union[SecondOrderTransformation, optax.GradientTransformation]
"""A type that can be used to represent either a first or second order transformation."""


def chain_second_order(*args: AnySecondOrderTransformation) -> SecondOrderTransformation:
    """Applies a list of chainable update transformations. Analogous to optax.chain,
    but for second order transformations.
    """

    init_fns = []
    update_fns = []
    update_hessian_fns: List[Optional[HessianUpdateFn]] = []

    for arg in args:
        if isinstance(arg, SecondOrderTransformation):
            init_fns.append(arg.init)
            update_fns.append(arg.update)
            update_hessian_fns.append(arg.update_hessian)
        else:
            init_fns.append(arg.init)
            update_fns.append(arg.update)
            update_hessian_fns.append(None)

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

    def update_hessian_fn(state, fn, model, *batch, **batch_kwargs):
        if len(update_hessian_fns) != len(state):
            raise ValueError(
                "The number of updates and states has to be the same in chain! Make sure you have called init first!"
            )

        new_state = []
        for s, update_fn in zip(state, update_hessian_fns):
            if update_fn is None:
                new_state.append(s)
            else:
                new_s = update_fn(s, fn, model, *batch, **batch_kwargs)
                new_state.append(new_s)
        return tuple(new_state)

    return SecondOrderTransformation(init_fn, update_fn, update_hessian_fn)


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
                dtype = _find_first_floating_dtype(numeric_hps)
            else:
                dtype = hyperparam_dtype
            hparams = {k: jnp.asarray(_convert_floats(v, dtype)) for k, v in numeric_hps.items()}
            hparams.update(schedule_fn(count, dtype))
            return InjectHyperparamsState(  # pylint:disable=too-many-function-args
                count, hparams, inner_factory(**other_hps, **hparams).init(params)
            )

        def update_fn(updates, state, params=None):
            if hyperparam_dtype is None:
                dtype = _find_first_floating_dtype(updates)
            else:
                dtype = hyperparam_dtype
            hparams = {k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(state.count, dtype))
            updates, inner_state = inner_factory(**other_hps, **hparams).update(updates, state.inner_state, params)
            # pylint:disable=too-many-function-args
            return updates, InjectHyperparamsState(state.count + 1, hparams, inner_state)
            # pylint:enable=too-many-function-args

        def _find_first_floating_dtype(updates):
            dtype = jnp.float32
            for v in jax.tree_util.tree_leaves(updates):
                if isinstance(v, jnp.ndarray):
                    if isinstance(v.dtype, jnp.floating):
                        dtype = v.dtype
                        break
            return dtype

        def update_hessian(state, fn, model, *batch, **batch_kwargs):
            if hyperparam_dtype is None:
                dtype = _find_first_floating_dtype(batch)
            else:
                dtype = hyperparam_dtype
            hparams = {k: _convert_floats(v, dtype) for k, v in state.hyperparams.items()}
            hparams.update(schedule_fn(state.count, dtype))
            new_inner_state = inner_factory(**other_hps, **hparams).update_hessian(
                state.inner_state,
                fn,
                model,
                *batch,
                **batch_kwargs,
            )

            # pylint:disable=too-many-function-args
            return InjectHyperparamsState(state.count, hparams, new_inner_state)
            # pylint:enable=too-many-function-args

        return SecondOrderTransformation(init_fn, update_fn, update_hessian)

    return wrapped_transform


def _convert_floats(x, dtype):
    """Convert float-like inputs to dtype, rest pass through."""
    if jax.dtypes.scalar_type_of(x) == float:
        return jnp.asarray(x, dtype=dtype)
    return x
