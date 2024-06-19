import asyncio
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.experimental.array_serialization.serialization as ser
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from levanter.utils import fsspec_utils


class JaggedArray(eqx.Module):
    """
    A jagged array is a collection of arrays of varying lengths.
    We represent this as a single array with an accompanying array of offsets.

    Note that JAX doesn't really support jagged arrays, so we have to be careful about how we use them.
    In particular, you shouldn't be using these directly in jitted code unless you're careful.

    Typically, we just use these for data loading, with the backing arrays being tensorstore arrays.
    """

    offsets: jax.Array | ts.TensorStore
    data: jax.Array | ts.TensorStore
    # shapes is probably premature architecture, but it amuses me.
    shapes: Optional[jax.Array | ts.TensorStore] = None  # (len(offsets), len(data.shape)-1)

    def __init__(
        self,
        offsets: jax.Array | ts.TensorStore,
        data: jax.Array | ts.TensorStore,
        shapes: Optional[jax.Array | ts.TensorStore] = None,
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


async def _ts_open_async(path: Optional[str], dtype: jnp.dtype, shape, *, mode):
    mode = _mode_to_open_mode(mode)
    if path is None:
        import uuid

        random_name = str(uuid.uuid4())
        spec = ts.Spec({"driver": "zarr", "kvstore": f"memory://{random_name}"})
    else:
        spec = ser.get_tensorstore_spec(path)
        fsspec_utils.mkdirs(os.path.dirname(path))

    # TODO: groups?
    # TODO: set chunk sizes
    return await ts.open(spec, dtype=jnp.dtype(dtype).name, shape=shape, **mode)


def _mode_to_open_mode(mode: str):
    if mode == "r":
        return {"read": True, "write": False}
    elif mode == "w":
        return {"read": True, "write": True, "delete_existing": True, "create": True}
    elif mode == "a":
        return {"read": True, "write": True, "delete_existing": False, "create": True}
    else:
        raise ValueError(f"Invalid mode: {mode}")


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

    offsets: ts.TensorStore
    data: ts.TensorStore
    shapes: Optional[ts.TensorStore]  # (len(offsets), len(data.shape)-1)
    item_rank: int = 1

    @staticmethod
    async def open_async(path: Optional[str], *, mode="a", item_rank=1, dtype):
        offset_path = _extend_path(path, "offsets")
        offsets = await _ts_open_async(offset_path, jnp.int64, [1], mode=mode)

        data_path = _extend_path(path, "data")
        data = await _ts_open_async(data_path, dtype, [0], mode=mode)

        if item_rank > 1:
            shape_path = _extend_path(path, "shapes")
            shapes = await _ts_open_async(shape_path, jnp.int64, [0, item_rank - 1], mode=mode)
        else:
            shapes = None

        return JaggedArrayBuilder(offsets, data, shapes, item_rank)

    @staticmethod
    def open(path: Optional[str], *, mode="a", item_rank=1, dtype):
        return asyncio.run(JaggedArrayBuilder.open_async(path, mode=mode, item_rank=item_rank, dtype=dtype))

    async def append_async(self, data: jax.Array):
        await self.extend_async([data])

    def append(self, data: jax.Array):
        asyncio.run(self.append_async(data))

    async def extend_async(self, arrays: Sequence[jax.Array]):
        if self.shapes is not None:
            for data in arrays:
                if data.ndim != self.item_rank:
                    raise ValueError(f"Expected data to have rank {self.item_rank}, got {data.ndim}")
            shapes = np.array([data.shape[:-1] for data in arrays], dtype=np.int64)
        else:
            for data in arrays:
                if data.ndim > 1:
                    raise ValueError(f"Expected data to have rank 1, got {data.ndim}")
            shapes = None

        new_offsets = np.array([data.size for data in arrays], dtype=np.int64)
        new_offsets = np.cumsum(new_offsets) + self.data.shape[0]

        data = np.concatenate([data.reshape(-1) for data in arrays])
        self.data = await _append_ts_async(self.data, data)

        self.offsets = await _append_ts_async(self.offsets, new_offsets)

        if self.shapes is not None:
            self.shapes = await _append_ts_async(self.shapes, shapes)

    def extend(self, arrays: Sequence[jax.Array]):
        asyncio.run(self.extend_async(arrays))

    def __len__(self):
        return self.offsets.shape[0] - 1

    async def get_item_async(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                raise ValueError("JaggedArrayBuilder doesn't support slicing with step != 1")
            shapes = None if self.shapes is None else self.shapes[start:stop]
            # NB: JaggedArray not JaggedArrayBuilder
            # TODO: use a transformed TS?
            offsets = self.offsets[start : stop + 1].read().result()
            data_start, data_stop = offsets[0], offsets[-1]
            new_offsets = offsets - offsets[0]
            return JaggedArray(new_offsets, await self.data[data_start:data_stop].read(), shapes)
        else:
            try:
                start, stop = await self.offsets[item : item + 2].read()
                data = await self.data[start:stop].read()

                if self.shapes is not None:
                    shapes = np.array(self.shapes[item])
                    data = data.reshape(*shapes, -1)
                return data
            except ValueError as e:
                # ts raises a value error for an index out of bounds OUT_OF_RANGE
                if "OUT_OF_RANGE" in str(e):
                    raise IndexError(f"JaggedArrayBuilder index out of range: {item}") from e
                else:
                    raise e

    def __getitem__(self, item):
        return asyncio.run(self.get_item_async(item))


async def _append_ts_async(store: ts.TensorStore, data: np.ndarray):
    out_store = await store.resize(exclusive_max=[store.shape[0] + data.shape[0], *store.shape[1:]])
    await out_store[out_store.shape[0] - data.shape[0] :].write(data)
    return out_store
