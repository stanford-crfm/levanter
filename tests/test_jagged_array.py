import math
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.newstore.jagged_array import JaggedArray, JaggedArrayStore


class TestJaggedArray:
    def test_construction(self):
        offsets = jnp.array([0, 3, 5])
        data = jnp.array([1, 2, 3, 4, 5])
        shapes = None

        jagged_array = JaggedArray(offsets, data, shapes)
        assert len(jagged_array) == 2
        assert jnp.all(jagged_array[0] == jnp.array([1, 2, 3]))
        assert jnp.all(jagged_array[1] == jnp.array([4, 5]))

    def test_slicing(self):
        offsets = jnp.array([0, 3, 5])
        data = jnp.array([1, 2, 3, 4, 5])
        shapes = None

        jagged_array = JaggedArray(offsets, data, shapes)
        sliced_array = jagged_array[1:]
        assert len(sliced_array) == 1
        assert jnp.all(sliced_array[0] == jnp.array([4, 5]))

    def test_slicing_with_shapes(self):
        offsets = jnp.array([0, 3, 5])
        data = jnp.array([1, 2, 3, 4, 5])
        shapes = jnp.array([[2], [1]])

        jagged_array = JaggedArray(offsets, data, shapes)
        sliced_array = jagged_array[1:]
        assert len(sliced_array) == 1
        assert jnp.all(sliced_array[0] == jnp.array([4, 5]))

    def test_empty_array(self):
        offsets = jnp.array([0])
        data = jnp.array([])
        jagged_array = JaggedArray(offsets, data)
        assert len(jagged_array) == 0
        with pytest.raises(IndexError):
            jagged_array[0]

    def test_single_element_array(self):
        offsets = jnp.array([0, 1])
        data = jnp.array([1])
        jagged_array = JaggedArray(offsets, data)
        assert len(jagged_array) == 1
        assert jnp.all(jagged_array[0] == jnp.array([1]))

    def test_out_of_bounds_index(self):
        offsets = jnp.array([0, 3, 5])
        data = jnp.array([1, 2, 3, 4, 5])
        shapes = None

        jagged_array = JaggedArray(offsets, data, shapes)
        with pytest.raises(IndexError):
            jagged_array[2]

    def test_step_slicing(self):
        offsets = jnp.array([0, 3, 5])
        data = jnp.array([1, 2, 3, 4, 5])
        shapes = None

        jagged_array = JaggedArray(offsets, data, shapes)
        with pytest.raises(ValueError):
            jagged_array[::2]


class TestJaggedArrayStore:
    def test_append_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            data2 = jnp.array([[5.0]])

            builder.append(data1)
            builder.append(data2)

            assert len(builder) == 2

            result1 = builder[0]
            assert jnp.all(result1 == data1)

            result2 = builder[1]
            assert jnp.all(result2 == data2)

            result_slice = builder[0:2]
            assert isinstance(result_slice, JaggedArray)

    def test_extend_with_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            data2 = jnp.array([[5.0]])

            builder.extend([data1, data2])

            assert len(builder) == 2

            result1 = builder[0]
            assert jnp.all(result1 == data1)

            result2 = builder[1]
            assert jnp.all(result2 == data2)

    def test_append_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.float32)
            with pytest.raises(ValueError):
                builder.append(jnp.array([[1.0, 2.0]]))

    def test_append_single_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=1, dtype=jnp.float32)

            data = jnp.array([1.0, 2.0, 3.0])
            builder.append(data)

            assert len(builder) == 1

            result = builder[0]
            assert jnp.all(result == data)

    def test_append_multi_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            data2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])

            builder.append(data1)
            builder.append(data2)

            assert len(builder) == 2

            result1 = builder[0]
            assert jnp.all(result1 == data1)

            result2 = builder[1]
            assert jnp.all(result2 == data2)

    def test_getitem_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            builder.append(data)

            with pytest.raises(IndexError):
                builder[2]

    def test_step_slicing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayStore.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            builder.append(data)

            with pytest.raises(ValueError):
                builder[::2]


async def create_builder_with_data(directory, num_sequences: int, sequence_length: int | tuple[int, ...]):
    if isinstance(sequence_length, int):
        sequence_length = (sequence_length,)

    """Helper function to create a JaggedArrayStore with specific data."""
    seed = jax.random.PRNGKey(num_sequences * math.prod(sequence_length))

    builder = await JaggedArrayStore.open_async(directory, item_rank=len(sequence_length), dtype=jnp.int64)
    for i in range(num_sequences):
        key, seed = jax.random.split(seed)
        data = jax.random.randint(key, sequence_length, 0, 100)
        await builder.append_async(data)

    return builder


def create_builder_with_data_sync(
    directory, num_sequences: int, sequence_length: int | tuple[int, ...]
) -> JaggedArrayStore:
    if isinstance(sequence_length, int):
        sequence_length = (sequence_length,)

    """Helper function to create a JaggedArrayStore with specific data."""
    seed = jax.random.PRNGKey(num_sequences * math.prod(sequence_length))

    builder = JaggedArrayStore.open(directory, item_rank=len(sequence_length), dtype=jnp.int64)
    for i in range(num_sequences):
        key, seed = jax.random.split(seed)
        data = jax.random.randint(key, sequence_length, 0, 100)
        builder.append(data)

    return builder


@pytest.mark.asyncio
async def test_trim_to_size_async():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)

    # Initial size
    initial_size = len(builder)
    assert initial_size == 10

    expected_data = list(await builder.get_item_async(slice(0, 10)))

    # Trim to smaller size
    await builder.trim_to_size_async(5)
    new_size = len(builder)
    assert new_size == 5

    # Verify the data integrity
    trimmed_data = await builder.data.read()
    assert jnp.all(trimmed_data == jnp.concatenate(expected_data[:5]))

    # Trim to zero size
    await builder.trim_to_size_async(0)
    new_size = len(builder)
    assert new_size == 0

    # Verify the data integrity
    trimmed_data = await builder.data.read()
    assert trimmed_data.size == 0


@pytest.mark.asyncio
async def test_trim_to_size_larger_than_current():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=1000)
    expected_data = list(await builder.get_item_async(slice(0, 10)))

    # Initial size
    initial_size = len(builder)
    assert initial_size == 10

    # Trim to a larger size than current (should not change)
    await builder.trim_to_size_async(15)
    new_size = len(builder)
    assert new_size == 10

    # Verify the data integrity
    trimmed_data = await builder.data.read()
    assert np.array_equal(trimmed_data, jnp.concatenate(expected_data[:10]))


@pytest.mark.asyncio
async def test_trim_to_size_with_shapes_async():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = await create_builder_with_data(tmpdir, num_sequences=10, sequence_length=(10, 100))
    expected_shapes = list(await builder.shapes.read())

    # Trim to smaller size
    await builder.trim_to_size_async(5)
    new_size = len(builder)
    assert new_size == 5

    # Verify the shapes integrity
    trimmed_shapes = await builder.shapes.read()
    assert np.array_equal(trimmed_shapes, jnp.stack(expected_shapes[:5]))


def test_trim_to_size():
    tmpdir = tempfile.TemporaryDirectory().name
    builder = create_builder_with_data_sync(tmpdir, num_sequences=10, sequence_length=1000)

    # Initial size
    initial_size = len(builder)
    assert initial_size == 10

    expected_data = list(builder[0:10])

    # Trim to smaller size
    builder.trim_to_size(5)
    new_size = len(builder)
    assert new_size == 5

    # Verify the data integrity
    trimmed_data = builder.data.read().result()
    assert jnp.all(trimmed_data == jnp.concatenate(expected_data[:5]))

    # Trim to zero size
    builder.trim_to_size(0)
    new_size = len(builder)
    assert new_size == 0

    # Verify the data integrity
    trimmed_data = builder.data.read().result()
    assert trimmed_data.size == 0


if __name__ == "__main__":
    pytest.main()
