import asyncio
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import fsspec.core
import jax
import jax.experimental.array_serialization.serialization as ser
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from levanter.utils import fsspec_utils
from levanter.utils.thread_utils import future_from_value


# zarr suggests 1MB chunk size (in bytes, but whatever)
# at 4 bytes this is 256k elements
DEFAULT_CHUNK_SIZE = 256 * 1024
DEFAULT_WRITE_CHUNK_SIZE = DEFAULT_CHUNK_SIZE * 512


@dataclass
class JaggedArrayStore:
    """
    A jagged array is a collection of arrays of varying lengths.
    We represent this as a single array with an accompanying array of offsets.

    Note that JAX doesn't really support jagged arrays, so we have to be careful about how we use them.
    Typically, we just use these for data loading.

    PERFORMANCE: accessing an individual row (or a single small slice of the underlying data) is very slow.
    Where ever possible, use get_batch to get multiple rows at once for as large a batch as possible.
    High latency, but high throughput.
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
        return int(self.offsets[0].read().result())

    async def num_rows_async(self):
        return int(await self.offsets[0].read())

    @property
    def data_size(self):
        return int(self.offsets[self.num_rows].read().result())

    async def append_async(self, data: jax.Array):
        await self.extend_async([data])

    def append(self, data: jax.Array):
        self.extend([data])

    async def trim_to_size_async(self, size: int):
        """
        Trims so we have exactly `size` rows in the jagged array.
        """
        if size >= len(self):
            return

        current_data_size = self.data_size
        current_num_rows = await self.num_rows_async()

        offsets_fut = self.offsets[size + 1 : current_num_rows + 1].write(0)

        if size == 0:
            new_max = 0
        else:
            new_max = int(await self.offsets[size].read())

        f1 = self.offsets[0].write(size)

        # Trim the shapes
        if self.shapes is not None:
            shape_fut = self.shapes[size:current_num_rows].write(
                np.zeros(self.shapes.shape[1:], dtype=self.shapes.dtype.name)
            )
        else:
            shape_fut = None

        data_fut = self.data[new_max:current_data_size].write(np.zeros((), dtype=self.data.dtype.name))
        await f1

        await shape_fut if shape_fut is not None else None
        await data_fut
        await offsets_fut

    def trim_to_size(self, size: int):
        if size >= self.num_rows:
            return

        old_len = len(self)
        old_data_size = self.data_size

        if self.shapes is not None:
            shape_fut = self.shapes[size:old_len].write(np.zeros(self.shapes.shape[1:], dtype=self.shapes.dtype.name))
        else:
            shape_fut = None

        f1 = self.offsets[0].write(size)

        if size == 0:
            new_max = 0
        else:
            new_max = int(self.offsets[size].read().result())
        data_fut = self.data[new_max:old_data_size].write(np.zeros((), dtype=self.data.dtype.name))

        f1.result()
        offsets_fut = self.offsets[size + 1 : old_data_size + 1].write(0)

        data_fut.result()
        offsets_fut.result()

        if shape_fut is not None:
            shape_fut.result()

    async def extend_async(self, arrays: Sequence[jax.Array]):
        data, new_offsets, shapes = self._prepare_batch(arrays)

        num_rows = await self.num_rows_async()
        num_added = len(arrays)
        current_data_size = self.data_size

        # Write to resized arrays concurrently, adjusting offsets explicitly
        write_tasks = [
            self.data[current_data_size : current_data_size + len(data)].write(data),
            self.offsets[num_rows + 1 : num_rows + num_added + 1].write(new_offsets),
        ]
        if self.shapes is not None:
            write_tasks.append(self.shapes[num_rows : num_rows + num_added].write(shapes))

        await asyncio.gather(*write_tasks)

        # Update num_rows
        int(self.offsets[self.num_rows].read().result())
        await self.offsets[0].write(num_rows + len(arrays))
        # print("done")

    def extend(self, arrays: Sequence[jax.Array]):
        data, new_offsets, shapes = self._prepare_batch(arrays)

        num_rows = self.num_rows
        num_added = len(arrays)
        current_data_size = self.data_size

        write_tasks = [
            self.data[current_data_size : current_data_size + len(data)].write(data),
            self.offsets[num_rows + 1 : num_rows + num_added + 1].write(new_offsets),
        ]
        if self.shapes is not None:
            write_tasks.append(self.shapes[num_rows : num_rows + num_added].write(shapes))

        for task in write_tasks:
            task.result()

        # Update num_rows. We want to make sure this comes after the other data is committed to avoid a race
        self.offsets[0].write(num_rows + len(arrays)).result()

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
            raise NotImplementedError("Slicing not supported")
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

    async def get_batch(self, indices: Sequence[int]) -> Sequence[jax.Array]:
        # get indices
        with ts.Batch():
            all_indices_futs = [self._bounds_for_rows_async(indices[i], indices[i] + 1) for i in range(len(indices))]

        # shapes, if applicable
        if self.shapes is not None:
            with ts.Batch():
                shapes_futs = [self.shapes[i].read() for i in indices]

        all_indices = [(start, stop) for start, stop, _ in await asyncio.gather(*all_indices_futs)]

        # get data
        with ts.Batch():
            data_futs = [self.data[start:stop].read() for start, stop in all_indices]

        data = await asyncio.gather(*data_futs)

        if self.shapes is not None:
            shapes = await asyncio.gather(*shapes_futs)

            data = [d.reshape(*s, -1) for d, s in zip(data, shapes)]

        return data

    def get_batch_sync(self, indices: Sequence[int]) -> Sequence[jax.Array]:
        all_indices = self._bounds_for_rows_batch(indices)

        with ts.Batch():
            # shapes, if applicable
            if self.shapes is not None:
                shapes_futs = [self.shapes[i].read() for i in indices]

            data_futs = [self.data[start:stop].read() for start, stop in all_indices]

        data = [d.result() for d in data_futs]

        if self.shapes is not None:
            shapes = [s.result() for s in shapes_futs]  # noqa
            data = [d.reshape(*s, -1) for d, s in zip(data, shapes)]

        return data

    def __getitem__(self, item):
        if isinstance(item, slice):
            # raise NotImplementedError("Slicing not supported")
            # # TODO: do we need to avoid reading len(self)?
            # start, stop, step = item.indices(len(self))
            # if step != 1:
            #     raise ValueError("JaggedArrayStore doesn't support slicing with step != 1")
            # shapes = None if self.shapes is None else self.shapes[start:stop]
            # # NB: JaggedArray not JaggedArrayStore
            # # TODO: use a transformed TS?
            # data_start, data_stop, offsets = self._bounds_for_rows(start, stop)
            # new_offsets = offsets - offsets[0]
            # return JaggedArray(new_offsets, self.data[data_start:data_stop].read().result(), shapes)
            start, stop, step = item.indices(len(self))
            # for now, just read the data into a list

            return [self[i] for i in range(start, stop, step)]
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
        num_rows = self.num_rows
        if start >= num_rows or stop > num_rows:
            raise IndexError("Index out of bounds")
        start, stop, step = slice(start, stop).indices(num_rows)
        offsets = self.offsets[start : stop + 1].read().result()
        data_start, data_stop = offsets[0], offsets[-1]
        if start == 0:
            # The first offset is the number of rows
            data_start = 0
            offsets[0] = 0

        return data_start, data_stop, offsets

    def _bounds_for_rows_batch(self, indices):
        num_rows = self.num_rows
        offsets_futs: list = []

        zero_pos = None

        with ts.Batch():
            for index in indices:
                if index >= num_rows or index < 0:
                    raise IndexError("Index out of bounds")
                offsets = self.offsets[index : index + 2].read()
                offsets_futs.append(offsets)

                if index == 0:
                    zero_pos = len(offsets_futs) - 1

        offsets = [fut.result() for fut in offsets_futs]
        offsets = [(offset[0], offset[-1]) for offset in offsets]

        if zero_pos is not None:
            offsets[zero_pos] = [0, offsets[zero_pos][1]]

        return offsets

    async def _bounds_for_rows_async(self, start, stop):
        offsets = await self.offsets[start : stop + 1].read()
        data_start, data_stop = offsets[0], offsets[-1]
        if start == 0:
            # The first offset is the number of rows
            data_start = 0
            offsets[0] = 0

        return data_start, data_stop, offsets


def _unshaped_spec(store: ts.TensorStore) -> ts.Spec:
    spec = store.spec(retain_context=True)
    return spec


def _ts_open_sync(path: Optional[str], dtype: jnp.dtype, shape, *, mode):
    spec = _get_spec(path, shape)
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
    try:
        return ts.open(
            spec,
            dtype=jnp.dtype(dtype).name,
            shape=[2**54, *shape[1:]],
            # chunk_layout=ts.ChunkLayout(
            #     read_chunk_shape=[DEFAULT_CHUNK_SIZE, *shape[1:]],
            #     write_chunk_shape=[DEFAULT_WRITE_CHUNK_SIZE, *shape[1:]]
            # ),
            # compression={"codec": "zstd", "compression_level": 5},
            **mode,
        ).result()
    except ValueError as e:
        if "NOT_FOUND" in str(e):
            raise FileNotFoundError(f"File not found: {path}") from e
        else:
            raise e


async def _ts_open_async(path: Optional[str], dtype: jnp.dtype, shape, *, mode):
    spec = _get_spec(path, shape)
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
        shape=[2**54, *shape[1:]],
        # chunk_layout=ts.ChunkLayout(
        #     read_chunk_shape=[DEFAULT_CHUNK_SIZE, *shape[1:]],
        #     write_chunk_shape=[DEFAULT_WRITE_CHUNK_SIZE, *shape[1:]]
        # ),
        # compression={"codec": "zstd", "compression_level": 5},
        **mode,
    )


def _get_spec(path, shape):
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
        store = spec.get("kvstore")
        spec = {"driver": "zarr3", "kvstore": store}
        fsspec_utils.mkdirs(os.path.dirname(path))
        spec["metadata"] = {
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [DEFAULT_WRITE_CHUNK_SIZE, *shape[1:]]},
            },
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [DEFAULT_CHUNK_SIZE, *shape[1:]],
                        "codecs": [{"name": "blosc", "configuration": {"clevel": 5}}],
                    },
                }
            ],
        }
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
