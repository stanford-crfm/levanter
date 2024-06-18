import os
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy
import numpy as np
import zarr


class JaggedArray(eqx.Module):
    """
    A jagged array is a collection of arrays of varying lengths.
    We represent this as a single array with an accompanying array of offsets.

    Note that JAX doesn't really support jagged arrays, so we have to be careful about how we use them.
    In particular, you shouldn't be using these directly in jitted code unless you're careful.

    Typically, we just use these for data loading, with the backing arrays being tensorstore arrays.
    """

    offsets: jax.Array | zarr.Array
    data: jax.Array | zarr.Array
    # shapes is probably premature architecture, but it amuses me.
    shapes: Optional[jax.Array | zarr.Array] = None  # (len(offsets), len(data.shape)-1)

    def __init__(
        self,
        offsets: jax.Array | zarr.Array,
        data: jax.Array | zarr.Array,
        shapes: Optional[jax.Array | zarr.Array] = None,
    ):
        self.offsets = offsets
        self.data = data
        self.shapes = shapes

    def __len__(self):
        return self.offsets.shape[0] - 1

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                raise ValueError("JaggedArray doesn't support slicing with step != 1")
            if stop > len(self):
                raise IndexError("JaggedArray index out of range")

            shapes = None if self.shapes is None else self.shapes[start:stop]
            new_offsets = self.offsets[start : stop + 1] - self.offsets[start]
            return JaggedArray(new_offsets, self.data[self.offsets[start] : self.offsets[stop]], shapes)
        else:
            if item >= len(self):
                raise IndexError("JaggedArray index out of range")
            data = self.data[self.offsets[item] : self.offsets[item + 1]]

            if self.shapes is not None:
                shapes = self.shapes[item]
                data = data.reshape(*shapes, -1)

            return data


def _zarr_open(path: Optional[str], dtype: jnp.dtype, shape):
    if path is None:
        return zarr.zeros(shape, dtype=dtype.name)

    # TODO: groups?
    # TODO: set chunk sizes
    return zarr.open_array(zarr.storage.FSStore(path), mode="a", shape=shape, dtype=jnp.dtype(dtype).name)


def _extend_path(path: Optional[str], extra: str):
    if path == "memory" or path is None:
        return path
    else:
        return os.path.join(path, extra)


@dataclass
class JaggedArrayBuilder:
    """
    A builder for jagged arrays.
    """

    offsets: zarr.Array
    data: zarr.Array
    shapes: Optional[zarr.Array] = None  # (len(offsets), len(data.shape)-1)

    @staticmethod
    def open(path: Optional[str], *, item_rank=1, dtype):
        offset_path = _extend_path(path, "offsets")
        offsets = _zarr_open(offset_path, jnp.int64, [1])

        data_path = _extend_path(path, "data")
        data = _zarr_open(data_path, dtype, [0])

        if item_rank > 1:
            shape_path = _extend_path(path, "shapes")
            shapes = _zarr_open(shape_path, jnp.int64, [0, item_rank - 1])
        else:
            shapes = None

        return JaggedArrayBuilder(offsets, data, shapes)

    def append(self, data: jax.Array):
        if self.shapes is not None:
            if data.ndim != self.shapes.shape[1] + 1:
                raise ValueError(f"Expected data to have rank {self.shapes.ndim + 1}, got {data.ndim}")

            shapes = np.array(data.shape[:-1], dtype=np.int64)
            self.shapes.append(numpy.expand_dims(shapes, axis=0))
        else:
            if data.ndim > 1:
                raise ValueError(f"Expected data to have rank 1, got {data.ndim}")

        data = data.reshape(-1)
        data = np.asarray(data, dtype=data.dtype)
        self.data.append(data)
        new_offset = self.data.shape[0]
        self.offsets.append(np.array([new_offset], dtype=np.int64))

    def __len__(self):
        return self.offsets.shape[0] - 1

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                raise ValueError("JaggedArrayBuilder doesn't support slicing with step != 1")
            shapes = None if self.shapes is None else self.shapes[start:stop]
            # NB: JaggedArray not JaggedArrayBuilder
            new_offsets = self.offsets[start : stop + 1] - self.offsets[start]
            return JaggedArray(new_offsets, self.data[self.offsets[start] : self.offsets[stop]], shapes)
        else:
            data = self.data[self.offsets[item] : self.offsets[item + 1]]

            if self.shapes is not None:
                shapes = np.array(self.shapes[item])
                data = data.reshape(*shapes, -1)

            return data
