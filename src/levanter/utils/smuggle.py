# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for smuggling values across JAX transformation boundaries.

This module provides :class:`Smuggler`, a small helper that is loosely inspired by
Flax's ``reap``/``sow`` helpers.  A :class:`Smuggler` keeps track of a thread-local
stack of containers.  Code that executes inside a JAX transform can populate the
active container (e.g. by calling ``Smuggler.get`` and mutating the returned
mapping).  The helper functions returned by :func:`Smuggler.wrap_transform`
return the populated container together with the original outputs, ensuring that
no tracers escape the transformation while still allowing the caller to recover
the side data afterwards.

The primary use case in Levanter is to collect metrics inside ``jit``/``grad``
transforms without relying on host callbacks.  A simple example is

.. code-block:: python

    metrics_smuggler = Smuggler(dict)

    def log(metrics: dict[str, jax.Array | float]) -> None:
        if metrics_smuggler.is_active:
            metrics_smuggler.get().update(metrics)

    @metrics_smuggler.wrap_transform(jax.jit)
    def compiled_loss(x):
        log({"loss": x})
        return x * 2

    metrics, value = compiled_loss(jnp.array(3.0))

``metrics`` now contains the dictionary produced inside the jitted function.
"""

from __future__ import annotations

import contextlib
import copy
import threading
from collections.abc import Callable, Iterator
from functools import update_wrapper, wraps
from typing import Any, Optional, TypeVar, Generic

from typing_extensions import ParamSpec


__all__ = [
    "Smuggler",
    "smugglify",
]


T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


MergeFn = Callable[[T, T], Optional[T]]


class _SmuggledCompiledFunction:
    """Wraps compiled JAX functions so their calls apply smuggler postprocessing."""

    def __init__(self, compiled: Any, postprocess: Callable[[Any], Any]) -> None:
        self._compiled = compiled
        self._postprocess = postprocess

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._compiled(*args, **kwargs)
        return self._postprocess(output)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        output = self._getattr("call")(*args, **kwargs)
        return self._postprocess(output)

    def unsafe_call(self, *args: Any, **kwargs: Any) -> Any:
        output = self._getattr("unsafe_call")(*args, **kwargs)
        return self._postprocess(output)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._compiled, item)

    def __dir__(self) -> list[str]:
        return sorted({*dir(self._compiled), *type(self).__dict__.keys()})

    def _getattr(self, name: str) -> Callable[..., Any]:
        attr = getattr(self._compiled, name, None)
        if attr is None:
            raise AttributeError(name)
        if not callable(attr):
            raise TypeError(f"Attribute {name!r} is not callable on compiled object")
        return attr


class _SmuggledLowered:
    """Wraps lowered JAX objects so their executions apply smuggler postprocessing."""

    def __init__(self, lowered: Any, postprocess: Callable[[Any], Any]) -> None:
        self._lowered = lowered
        self._postprocess = postprocess

    def call(self, *args: Any, **kwargs: Any) -> Any:
        output = self._getattr("call")(*args, **kwargs)
        return self._postprocess(output)

    def compile(self, *args: Any, **kwargs: Any) -> _SmuggledCompiledFunction:
        compiled = self._getattr("compile")(*args, **kwargs)
        return _SmuggledCompiledFunction(compiled, self._postprocess)

    def unsafe_call(self, *args: Any, **kwargs: Any) -> Any:
        output = self._getattr("unsafe_call")(*args, **kwargs)
        return self._postprocess(output)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = getattr(self._lowered, "__call__", None)
        if call is None:
            raise TypeError("Lowered object is not directly callable")
        output = call(*args, **kwargs)
        return self._postprocess(output)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._lowered, item)

    def __dir__(self) -> list[str]:
        return sorted({*dir(self._lowered), *type(self).__dict__.keys()})

    def _getattr(self, name: str) -> Callable[..., Any]:
        attr = getattr(self._lowered, name, None)
        if attr is None:
            raise AttributeError(name)
        if not callable(attr):
            raise TypeError(f"Attribute {name!r} is not callable on lowered object")
        return attr


class _SmuggledWrappedFunction:
    """Callable proxy that forwards attribute access to the transformed function."""

    _forwarded_methods = {"lower"}

    def __init__(
        self,
        original: Callable[P, Any],
        transformed: Callable[P, Any],
        postprocess: Callable[[Any], Any],
    ) -> None:
        self._original = original
        self._transformed = transformed
        self._postprocess = postprocess
        update_wrapper(self, original)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        output = self._transformed(*args, **kwargs)
        return self._postprocess(output)

    def __getattr__(self, item: str) -> Any:
        target = getattr(self._transformed, item)
        if item in self._forwarded_methods and callable(target):
            return self._wrap_forwarded_method(item, target)
        return target

    def __dir__(self) -> list[str]:
        own_attrs = set(super().__dir__())
        return sorted(own_attrs | set(dir(self._transformed)))

    def _wrap_forwarded_method(self, name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        if name == "lower":

            @wraps(method)
            def lowered(*args: Any, **kwargs: Any) -> _SmuggledLowered:
                result = method(*args, **kwargs)
                return _SmuggledLowered(result, self._postprocess)

            return lowered
        return method


class Smuggler(Generic[T]):
    """Thread-local helper used to capture side-channel data from JAX code.

    A :class:`Smuggler` manages a stack of containers (one per active context).
    Containers are created by ``factory`` and merged into their parent context
    when the context exits.  ``merge`` should mutate the parent container in
    place or return a replacement object.  If ``merge`` returns ``None`` the
    parent container is assumed to have been updated in place.

    The class exposes two core concepts:

    * ``activate``: A context manager that pushes a new container on the stack.
      The container is yielded to the caller so that they can mutate it directly.
    * ``wrap_transform`` / ``smugglify``: Utilities that activate a smuggler
      context around a JAX transform and return the populated container together
      with the original outputs.
    """

    def __init__(self, factory: Callable[[], T], *, merge: Optional[MergeFn[T]] = None):
        self._factory = factory
        self._merge: MergeFn[T] = merge or self._default_merge
        self._local = threading.local()

    # ------------------------------------------------------------------
    # stack management helpers
    # ------------------------------------------------------------------
    def _stack(self) -> list[T]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            setattr(self._local, "stack", stack)
        return stack

    @property
    def is_active(self) -> bool:
        """Returns ``True`` if the smuggler currently has an active context."""

        stack = getattr(self._local, "stack", None)
        return bool(stack)

    @contextlib.contextmanager
    def activate(self) -> Iterator[T]:
        """Activates the smuggler and yields a fresh container.

        The container is created by calling ``factory``.  When the context exits
        the container is merged into the parent context (if any).
        """

        stack = self._stack()
        payload = self._factory()
        stack.append(payload)
        try:
            yield payload
        finally:
            stack.pop()
            if stack:
                parent = stack[-1]
                merged = self._merge(parent, payload)
                if merged is not None and merged is not parent:
                    stack[-1] = merged

    def get(self) -> T:
        """Returns the container associated with the innermost active context."""

        stack = getattr(self._local, "stack", None)
        if not stack:
            raise RuntimeError("Smuggler context is not active")
        return stack[-1]

    def set(self, value: T) -> None:
        """Replaces the container associated with the innermost context."""

        stack = getattr(self._local, "stack", None)
        if not stack:
            raise RuntimeError("Smuggler context is not active")
        stack[-1] = value

    def update(self, transform: Callable[[T], Optional[T]]) -> None:
        """Applies ``transform`` to the current container.

        ``transform`` can either mutate the container in place and return
        ``None`` or return a replacement object, in which case the smuggler will
        install the returned value as the new container.
        """

        current = self.get()
        maybe_new = transform(current)
        if maybe_new is not None:
            self.set(maybe_new)

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def smugglify(
        self,
        transform: Callable[..., Callable[P, Any]],
        /,
        *transform_args: Any,
        postprocess: Optional[Callable[[Any], tuple[T, R]]] = None,
        **transform_kwargs: Any,
    ) -> Callable[[Callable[P, R]], Callable[P, tuple[T, R]]]:
        """Wraps ``transform`` so the returned function also yields the captured payload."""

        def decorator(fn: Callable[P, R]) -> Callable[P, tuple[T, R]]:
            def fn_with_metrics(*args: P.args, **kwargs: P.kwargs) -> tuple[R, T]:
                with self.activate() as payload:
                    result = fn(*args, **kwargs)
                    exported = self._export(payload)
                return result, exported

            transformed = transform(fn_with_metrics, *transform_args, **transform_kwargs)

            if postprocess is None:

                def _post(output: Any) -> tuple[T, R]:
                    result, metrics = output
                    return metrics, result

                current_postprocess: Callable[[Any], tuple[T, R]] = _post
            else:
                current_postprocess = postprocess

            return _SmuggledWrappedFunction(fn, transformed, current_postprocess)  # type: ignore[return-value]

        return decorator

    # ------------------------------------------------------------------
    # merge helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _default_merge(parent: T, child: T) -> Optional[T]:
        """Fallback merge strategy used when none is provided.

        The default implementation handles mutable mappings by invoking
        ``update``.  For other container types a ``merge`` callable must be
        supplied when constructing the smuggler.
        """

        if hasattr(parent, "update") and callable(getattr(parent, "update")):
            parent.update(child)  # type: ignore[attr-defined]
            return None
        raise TypeError("Smuggler requires a merge function for container type " f"{type(parent).__name__!r}.")

    @staticmethod
    def _export(payload: T) -> T:
        """Returns a safe-to-escape version of ``payload``.

        By default we attempt to create a shallow copy of the payload so that
        the original container (which might still contain tracer references) is
        not the object that escapes the JAX transformation.  If copying fails we
        simply return the payload itself.
        """

        try:
            return copy.copy(payload)
        except Exception:  # pragma: no cover - defensive guardrail
            return payload


def smugglify(
    transform: Callable[..., Callable[P, Any]],
    /,
    *default_transform_args: Any,
    smuggler: Optional[Smuggler[T]] = None,
    postprocess: Optional[Callable[[Any], tuple[T, R]]] = None,
    **default_transform_kwargs: Any,
) -> Callable[..., Any]:
    """Creates a smuggling wrapper around ``transform``.

    ``postprocess`` has the same meaning as in :meth:`Smuggler.smugglify` and is
    forwarded to each invocation.
    """

    chosen_smuggler: Smuggler[T]
    if smuggler is None:
        chosen_smuggler = Smuggler(dict)  # type: ignore[arg-type]
    else:
        chosen_smuggler = smuggler

    def wrapper(
        fn: Callable[P, R] | None = None,
        /,
        *transform_args: Any,
        smuggler: Optional[Smuggler[T]] = None,
        postprocess: Optional[Callable[[Any], tuple[T, R]]] = postprocess,
        **transform_kwargs: Any,
    ) -> Callable[..., Any]:
        active_smuggler = smuggler or chosen_smuggler

        def apply(target: Callable[P, R]) -> Callable[P, tuple[T, R]]:
            return active_smuggler.smugglify(
                transform,
                *(default_transform_args + transform_args),
                postprocess=postprocess,
                **({**default_transform_kwargs, **transform_kwargs}),
            )(target)

        if fn is None:
            return apply
        if not callable(fn):  # pragma: no cover - defensive programming
            raise TypeError("First argument to smugglify must be callable or None")
        return apply(fn)

    return wrapper
