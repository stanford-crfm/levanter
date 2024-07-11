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


# zarr suggests 1MB chunk size (in bytes, but whatever)
# at 4 bytes this is
DEFAULT_CHUNK_SIZE = 256 * 1024


def _ts_open_sync(path: Optional[str], dtype: jnp.dtype, shape, *, mode):
    spec = _get_spec(path)
    mode = _mode_to_open_mode(mode)

    # Basically, we want to load the existing shape metadata if it exists
    if not mode.get("delete_existing", False):
        try:
            return ts.open(spec, **mode).result()
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
        chunk_layout=ts.ChunkLayout(chunk_shape=[DEFAULT_CHUNK_SIZE, *shape[1:]]),
        **mode,
    ).result()


async def _ts_open_async(path: Optional[str], dtype: jnp.dtype, shape, *, mode):
    spec = _get_spec(path)
    mode = _mode_to_open_mode(mode)

    # Basically, we want to load the existing shape metadata if it exists
    if not mode.get("delete_existing", False):
        try:
            return await ts.open(spec, **mode)
        except FileNotFoundError:
            pass
        except ValueError:
            pass

    # TODO: groups?
    # TODO: set chunk sizes
    return await ts.open(
        spec,
        dtype=jnp.dtype(dtype).name,
        shape=shape,
        chunk_layout=ts.ChunkLayout(chunk_shape=[DEFAULT_CHUNK_SIZE, *shape[1:]]),
        **mode,
    )


def _get_spec(path):
    if path is None:
        import uuid

        random_name = str(uuid.uuid4())
        spec = ts.Spec({"driver": "zarr", "kvstore": f"memory://{random_name}"})
    else:
        # make path absolute if it's not already
        protocol, _ = fsspec.core.split_protocol(path)
        if protocol is None:
            path = os.path.abspath(path)
        spec = ser.get_tensorstore_spec(path, ocdbt=False)
        fsspec_utils.mkdirs(os.path.dirname(path))
    return spec


def _mode_to_open_mode(mode: str):
    if mode == "r":
        return {"open_mode": ts.OpenMode(open=True)}
    elif mode == "w":
        return {"open_mode": ts.OpenMode(create=True, delete_existing=True)}
    elif mode == "a":
        return {"open_mode": ts.OpenMode(create=True, open=True, delete_existing=False)}
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _extend_path(path: Optional[str], extra: str):
    if path == "memory" or path is None:
        return path
    else:
        return os.path.join(path, extra)


@dataclass
class JaggedArrayStore:
    """
    A jagged array is a collection of arrays of varying lengths.
    We represent this as a single array with an accompanying array of offsets.

    Note that JAX doesn't really support jagged arrays, so we have to be careful about how we use them.
    Typically, we just use these for data loading.
    """

    offsets: ts.TensorStore  # offsets of the start of each array, except that index[0] is the number of arrays
    data: ts.TensorStore
    shapes: Optional[ts.TensorStore]  # (len(offsets), len(data.shape)-1)
    item_rank: int = 1

    @staticmethod
    async def open_async(path: Optional[str], *, mode="a", item_rank=1, dtype) -> "JaggedArrayStore":
        offset_path = _extend_path(path, "offsets")
        offsets = _ts_open_async(offset_path, jnp.int64, [1], mode=mode)

        data_path = _extend_path(path, "data")
        data = _ts_open_async(data_path, dtype, [0], mode=mode)

        if item_rank > 1:
            shape_path = _extend_path(path, "shapes")
            shapes = _ts_open_async(shape_path, jnp.int64, [0, item_rank - 1], mode=mode)
        else:
            shapes = None

        return JaggedArrayStore(await offsets, await data, await shapes if shapes is not None else None, item_rank)

    @staticmethod
    def open(path: Optional[str], *, mode="a", item_rank=1, dtype) -> "JaggedArrayStore":
        offset_path = _extend_path(path, "offsets")
        offsets = _ts_open_sync(offset_path, jnp.int64, [1], mode=mode)

        data_path = _extend_path(path, "data")
        data = _ts_open_sync(data_path, dtype, [0], mode=mode)

        if item_rank > 1:
            shape_path = _extend_path(path, "shapes")
            shapes = _ts_open_sync(shape_path, jnp.int64, [0, item_rank - 1], mode=mode)
        else:
            shapes = None

        return JaggedArrayStore(offsets, data, shapes, item_rank)

    @property
    def num_rows(self):
        return self.offsets[0].read().result()

    async def num_rows_async(self):
        return await self.offsets[0].read()

    @property
    def data_size(self):
        return self.offsets[self.num_rows].read().result()

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

        if size == 0:
            # Just reset the data
            new_data = self.data.resize(exclusive_max=[0])
            if self.shapes is not None:
                new_shapes = self.shapes.resize(exclusive_max=[0, *self.shapes.shape[1:]])
            else:
                new_shapes = None

            await self.offsets[0].write(0)
            new_offsets = self.offsets.resize(exclusive_max=[1])

            self.data = await new_data
            self.shapes = await new_shapes if new_shapes is not None else None
            self.offsets = await new_offsets
            return

        offsets = await self.offsets.resize(exclusive_max=[size + 1])
        new_max = await offsets[size].read()
        data = self.data.resize(exclusive_max=[new_max])

        f1 = offsets[0].write(size)

        # Trim the shapes
        if self.shapes is not None:
            shapes = self.shapes.resize(exclusive_max=[size, *self.shapes.shape[1:]])
            self.shapes = await shapes

        self.offsets = offsets
        self.data = await data
        await f1

    def trim_to_size(self, size: int):
        if size >= self.data_size:
            return

        if size == 0:
            # Just reset the data
            new_data = self.data.resize(exclusive_max=[0])
            if self.shapes is not None:
                new_shapes = self.shapes.resize(exclusive_max=[0, *self.shapes.shape[1:]])
            else:
                new_shapes = None

            self.offsets[0].write(0).result()
            new_offsets = self.offsets.resize(exclusive_max=[1])

            self.data = new_data.result()
            self.shapes = new_shapes.result() if new_shapes is not None else None
            self.offsets = new_offsets.result()
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
        f1 = self.offsets[0].write(size)
        self.data = data.result()
        if self.shapes is not None:
            self.shapes = shapes.result()

        f1.result()

    async def extend_async(self, arrays: Sequence[jax.Array]):
        data, new_offsets, shapes = self._prepare_batch(arrays)
        self.data = await _append_ts_async(self.data, data)

        self.offsets = await _append_ts_async(self.offsets, new_offsets)

        if self.shapes is not None:
            self.shapes = await _append_ts_async(self.shapes, shapes)

        await self.offsets[0].write(await self.offsets[0].read() + len(arrays))

    def extend(self, arrays: Sequence[jax.Array]):
        data, new_offsets, shapes = self._prepare_batch(arrays)
        current_len = self.offsets[0].read()
        self.data = _append_ts_sync(self.data, data)

        self.offsets = _append_ts_sync(self.offsets, new_offsets)

        if self.shapes is not None:
            self.shapes = _append_ts_sync(self.shapes, shapes)

        self.offsets[0].write(current_len.result() + len(arrays)).result()

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
        new_offsets = np.cumsum(new_offsets) + self.data_size
        data = np.concatenate([data.reshape(-1) for data in arrays])
        return data, new_offsets, shapes

    async def reload_async(self) -> "JaggedArrayStore":
        """
        Calls `resolve` on the underlying tensorstore objects, updating size information

        @return: new JaggedArrayStore with resolved tensorstores
        """
        offsets = ts.open(_unshaped_spec(self.offsets))
        data = ts.open(_unshaped_spec(self.data))
        shapes = future_from_value(None) if self.shapes is None else ts.open(_unshaped_spec(self.shapes.spec()))

        offsets, data, shapes = await asyncio.gather(offsets, data, shapes)

        return JaggedArrayStore(offsets, data, shapes, self.item_rank)

    def reload(self) -> "JaggedArrayStore":
        offsets = ts.open(_unshaped_spec(self.offsets))
        data = ts.open(_unshaped_spec(self.data))
        shapes = None if self.shapes is None else ts.open(_unshaped_spec(self.shapes.spec())).result()

        offsets = offsets.result()
        data = data.result()

        return JaggedArrayStore(offsets, data, shapes, self.item_rank)

    def __len__(self):
        return self.num_rows

    async def get_item_async(self, item):
        if isinstance(item, slice):
            len_self = await self.num_rows_async()
            start, stop, step = item.indices(len_self)
            if step != 1:
                raise ValueError("JaggedArrayStore doesn't support slicing with step != 1")
            shapes = None if self.shapes is None else self.shapes[start:stop]
            # NB: JaggedArray not JaggedArrayStore
            # TODO: use a transformed TS?
            data_start, data_stop, offsets = await self._bounds_for_rows_async(start, stop)
            new_offsets = offsets - offsets[0]
            return JaggedArray(new_offsets, await self.data[data_start:data_stop].read(), shapes)
        else:
            try:
                start, stop, _ = await self._bounds_for_rows_async(item, item + 1)
                data = await self.data[start:stop].read()

                if self.shapes is not None:
                    shapes = np.array(self.shapes[item])
                    data = data.reshape(*shapes, -1)
                return data
            except ValueError as e:
                # ts raises a value error for an index out of bounds OUT_OF_RANGE
                if "OUT_OF_RANGE" in str(e):
                    raise IndexError(f"JaggedArrayStore index out of range: {item}") from e
                else:
                    raise e

    def __getitem__(self, item):
        if isinstance(item, slice):
            # TODO: do we need to avoid reading len(self)?
            start, stop, step = item.indices(len(self))
            if step != 1:
                raise ValueError("JaggedArrayStore doesn't support slicing with step != 1")
            shapes = None if self.shapes is None else self.shapes[start:stop]
            # NB: JaggedArray not JaggedArrayStore
            # TODO: use a transformed TS?
            data_start, data_stop, offsets = self._bounds_for_rows(start, stop)
            new_offsets = offsets - offsets[0]
            return JaggedArray(new_offsets, self.data[data_start:data_stop].read().result(), shapes)
        else:
            try:
                start, stop, _ = self._bounds_for_rows(item, item + 1)
                data = self.data[start:stop].read().result()

                if self.shapes is not None:
                    shapes = np.array(self.shapes[item])
                    data = data.reshape(*shapes, -1)
                return data
            except ValueError as e:
                # ts raises a value error for an index out of bounds OUT_OF_RANGE
                if "OUT_OF_RANGE" in str(e):
                    raise IndexError(f"JaggedArrayStore index out of range: {item}") from e
                else:
                    raise e

    def _bounds_for_rows(self, start, stop):
        offsets = self.offsets[start : stop + 1].read().result()
        data_start, data_stop = offsets[0], offsets[-1]
        if start == 0:
            # The first offset is the number of rows
            data_start = 0
            offsets[0] = 0

        return data_start, data_stop, offsets

    async def _bounds_for_rows_async(self, start, stop):
        offsets = await self.offsets[start : stop + 1].read()
        data_start, data_stop = offsets[0], offsets[-1]
        if start == 0:
            # The first offset is the number of rows
            data_start = 0
            offsets[0] = 0

        return data_start, data_stop, offsets


# TODO: ideally we would grow the store in larger chunks, but in practice we write very large batches
# so this is probably fine.


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
