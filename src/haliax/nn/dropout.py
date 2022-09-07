from typing import Optional

import equinox as eqx
import jax

import haliax
from haliax.core import AxisSpec, NamedArray
from haliax.util import ensure_tuple


class Dropout(eqx.Module):
    """Applies dropout.

    Attributes:
        pdrop: The fraction of entries to set to zero.
        broadcast_axes: The dimensions to broadcast the dropout mask over. If set, these axes will share the same mask
    """

    # key difference from equinox: these are static fields
    pdrop: float = eqx.static_field()
    broadcast_axes: Optional[AxisSpec] = eqx.static_field()

    def __init__(
        self,
        pdrop: float = 0.5,
        broadcast_axes: Optional[AxisSpec] = None,
    ):
        self.pdrop = pdrop
        self.broadcast_axes = broadcast_axes

    def __call__(
        self,
        x: NamedArray,
        *,
        inference: bool,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> NamedArray:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to dropout.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
            `False` then it will take priority over `self.inference`. If `None`
            then the value from `self.inference` will be used.
        - `deterministic`: Deprecated alternative to `inference`.
        """

        if isinstance(self.pdrop, (int, float)) and self.pdrop == 0:
            return x
        elif self.pdrop == 1:
            return haliax.zeros_like(x)
        elif inference:
            return x
        elif key is None:
            raise RuntimeError("Dropout requires a key when running in non-deterministic mode.")
        else:
            with jax.named_scope(name="dropout"):

                if self.broadcast_axes is None:
                    if isinstance(x, NamedArray):
                        shape_to_generate = x.axes
                    else:
                        shape_to_generate = x.shape
                else:
                    axes = ensure_tuple(self.broadcast_axes)
                    shape_to_generate = tuple(ax for ax in x.axes if ax not in axes)

                q = 1 - self.pdrop
                q = x.dtype.type(q)
                mask = haliax.random.bernoulli(key, q, shape_to_generate)

                out = haliax.where(mask, x / q, x.dtype.type(0))  # type: ignore
                assert out.dtype == x.dtype
                return out
