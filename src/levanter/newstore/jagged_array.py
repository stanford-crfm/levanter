import asyncio
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import equinox as eqx
import fsspec.core
import jax
import jax.experimental.array_serialization.serialization as ser
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from levanter.utils import fsspec_utils
from levanter.utils.py_utils import future_from_value


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


def _ts_open_core(path: Optional[str], dtype: jnp.dtype, shape, *, mode):
    mode = _mode_to_open_mode(mode)
    if path is None:
        import uuid

        random_name = str(uuid.uuid4())
        spec = ts.Spec({"driver": "zarr", "kvstore": f"memory://{random_name}"})
    else:
        # make path absolute if it's not already
        protocol, _ = fsspec.core.split_protocol(path)
        if protocol is None:
            path = os.path.abspath(path)
        spec = ser.get_tensorstore_spec(path, ocdbt=True)
        fsspec_utils.mkdirs(os.path.dirname(path))

    # TODO: this is a bit of a hack
    # Basically, we want to load the existing shape metadata if it exists
    if not mode.get("delete_existing", False):
        try:
            return ts.open(spec, **mode).result()  # TODO: use async when applicable
        except FileNotFoundError:
            pass
        except ValueError:
            pass

    # TODO: groups?
    # TODO: set chunk sizes
    return ts.open(
        spec,
        dtype=jnp.dtype(dtype).name,
        shape=shape,
        chunk_layout=ts.ChunkLayout(chunk_shape=[8192, *shape[1:]]),
        **mode,
    )


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
        offsets, data, shapes = JaggedArrayBuilder._open_core(path, mode=mode, item_rank=item_rank, dtype=dtype)
        return JaggedArrayBuilder(await offsets, await data, await shapes if shapes is not None else shapes, item_rank)

    @staticmethod
    def open(path: Optional[str], *, mode="a", item_rank=1, dtype):
        offsets, data, shapes = JaggedArrayBuilder._open_core(path, mode=mode, item_rank=item_rank, dtype=dtype)
        return JaggedArrayBuilder(
            offsets.result(), data.result(), shapes.result() if shapes is not None else shapes, item_rank
        )

    @staticmethod
    def _open_core(path: Optional[str], *, mode="a", item_rank=1, dtype):
        offset_path = _extend_path(path, "offsets")
        offsets = _ts_open_core(offset_path, jnp.int64, [1], mode=mode)

        data_path = _extend_path(path, "data")
        data = _ts_open_core(data_path, dtype, [0], mode=mode)

        if item_rank > 1:
            shape_path = _extend_path(path, "shapes")
            shapes = _ts_open_core(shape_path, jnp.int64, [0, item_rank - 1], mode=mode)
        else:
            shapes = None

        return (offsets, data, shapes)

    @property
    def data_size(self):
        return self.offsets[self.offsets.shape[0] - 1].read().result()

    async def append_async(self, data: jax.Array):
        await self.extend_async([data])

    def append(self, data: jax.Array):
        asyncio.run(self.append_async(data))

    async def trim_to_size_async(self, size: int):
        """
        Trims so we have exactly `size` rows in the jagged array.
        """
        if size >= len(self):
            return

        # Trim the offsets
        offsets = await self.offsets.resize(exclusive_max=[size + 1])
        # Trim the data
        new_max = await offsets[size].read()
        data = self.data.resize(exclusive_max=[new_max])

        # Trim the shapes
        if self.shapes is not None:
            shapes = self.shapes.resize(exclusive_max=[size, *self.shapes.shape[1:]])
            self.shapes = await shapes

        self.offsets = offsets
        self.data = await data

    def trim_to_size(self, size: int):
        if size >= self.data_size:
            return

        if self.shapes is not None:
            shapes = self.shapes.resize(exclusive_max=[size, *self.shapes.size[1:]])

        # Trim the offsets
        offsets = self.offsets.resize(exclusive_max=[size + 1]).result()
        # Trim the data
        new_max = offsets[size].read().result()
        data = self.data.resize(exclusive_max=[new_max])

        # Trim the shapes

        self.offsets = offsets
        self.data = data.result()
        self.shapes = shapes.result()

    async def extend_async(self, arrays: Sequence[jax.Array]):
        data, new_offsets, shapes = self._prepare_batch(arrays)
        self.data = await _append_ts_async(self.data, data)

        self.offsets = await _append_ts_async(self.offsets, new_offsets)

        if self.shapes is not None:
            self.shapes = await _append_ts_async(self.shapes, shapes)

    def extend(self, arrays: Sequence[jax.Array]):
        data, new_offsets, shapes = self._prepare_batch(arrays)
        self.data = _append_ts_sync(self.data, data)

        self.offsets = _append_ts_sync(self.offsets, new_offsets)

        if self.shapes is not None:
            self.shapes = _append_ts_sync(self.shapes, shapes)

    def _prepare_batch(self, arrays):
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
        return data, new_offsets, shapes

    async def reload_async(self) -> "JaggedArrayBuilder":
        """
        Calls `resolve` on the underlying tensorstore objects, updating size information

        @return: new JaggedArrayBuilder with resolved tensorstores
        """
        # offsets = await self.offsets.resolve(fix_resizable_bounds=True)
        # data = await self.data.resolve(fix_resizable_bounds=True)
        # if self.shapes is not None:
        #     shapes = await self.shapes.resolve(fix_resizable_bounds=True)
        # else:
        #     shapes = None
        offsets = ts.open(_unshaped_spec(self.offsets))
        data = ts.open(_unshaped_spec(self.data))
        shapes = future_from_value(None) if self.shapes is None else ts.open(_unshaped_spec(self.shapes.spec()))

        offsets, data, shapes = await asyncio.gather(offsets, data, shapes)

        return JaggedArrayBuilder(offsets, data, shapes, self.item_rank)

    def reload(self) -> "JaggedArrayBuilder":
        offsets = ts.open(_unshaped_spec(self.offsets))
        data = ts.open(_unshaped_spec(self.data))
        shapes = None if self.shapes is None else ts.open(_unshaped_spec(self.shapes.spec())).result()

        offsets = offsets.result()
        data = data.result()

        return JaggedArrayBuilder(offsets, data, shapes, self.item_rank)

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
            return JaggedArray(new_offsets, self.data[data_start:data_stop].read().result(), shapes)
        else:
            try:
                start, stop = self.offsets[item : item + 2].read().result()
                data = self.data[start:stop].read().result()

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


async def _append_ts_async(store: ts.TensorStore, data: np.ndarray):
    out_store = await store.resize(exclusive_max=[store.shape[0] + data.shape[0], *store.shape[1:]])
    await out_store[out_store.shape[0] - data.shape[0] :].write(data)
    return out_store


def _append_ts_sync(store: ts.TensorStore, data: np.ndarray):
    out_store = store.resize(exclusive_max=[store.shape[0] + data.shape[0], *store.shape[1:]]).result()
    out_store[out_store.shape[0] - data.shape[0] :].write(data).result()
    return out_store


def _unshaped_spec(store: ts.TensorStore) -> ts.Spec:
    spec = store.spec(retain_context=True)
    return spec
