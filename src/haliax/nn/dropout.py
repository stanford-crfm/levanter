from typing import Optional

import equinox as eqx
import jax

import haliax
from haliax.core import NamedArray
from haliax.types import AxisSpec
from haliax.util import ensure_tuple


def dropout(x, pdrop, broadcast_axes=None, *, inference, key=None):
    """Applies dropout.

    **Arguments:**

    - `x`: An any-dimensional JAX array to dropout.
    - `pdrop`: The fraction of entries to set to zero.
    - `broadcast_axes`: The dimensions to broadcast the dropout mask over. If set, these axes will share the same mask
    - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
        which elements to dropout. (Keyword only argument.)
    - `inference`: As per [`equinox.nn.Dropout.__init__`][]. If `True` or
        `False` then it will take priority over `self.inference`. If `None`
        then the value from `self.inference` will be used.
    """
    if inference:
        return x
    elif isinstance(pdrop, (int, float)) and pdrop == 0:
        return x
    elif isinstance(pdrop, (int, float)) and pdrop == 1:
        return haliax.zeros_like(x)
    elif key is None:
        raise RuntimeError("Dropout requires a key when running in non-deterministic mode.")
    else:
        with jax.named_scope(name="dropout"):
            if broadcast_axes is None:
                if isinstance(x, NamedArray):
                    shape_to_generate = x.axes
                else:
                    shape_to_generate = x.shape
            else:
                axes = ensure_tuple(broadcast_axes)
                shape_to_generate = tuple(ax for ax in x.axes if ax not in axes)

            q = 1 - pdrop
            mask = haliax.random.bernoulli(key, shape_to_generate, q)
            q = x.dtype.type(q)

            out = haliax.where(mask, x / q, 0)
            assert out.dtype == x.dtype
            return out


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
        return dropout(
            x,
            self.pdrop,
            broadcast_axes=self.broadcast_axes,
            inference=inference,
            key=key,
        )
