import tempfile

import jax.numpy as jnp
import pytest

from levanter.newstore.jagged_array import JaggedArray, JaggedArrayBuilder


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


class TestJaggedArrayBuilder:
    def test_append_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayBuilder.open(tmpdir, item_rank=2, dtype=jnp.float32)

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

    def test_append_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayBuilder.open(tmpdir, item_rank=1, dtype=jnp.float32)
            with pytest.raises(ValueError):
                builder.append(jnp.array([[1.0, 2.0]]))

    def test_append_single_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayBuilder.open(tmpdir, item_rank=1, dtype=jnp.float32)

            data = jnp.array([1.0, 2.0, 3.0])
            builder.append(data)

            assert len(builder) == 1

            result = builder[0]
            assert jnp.all(result == data)

    def test_append_multi_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayBuilder.open(tmpdir, item_rank=2, dtype=jnp.float32)

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
            builder = JaggedArrayBuilder.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            builder.append(data)

            with pytest.raises(IndexError):
                builder[2]

    def test_step_slicing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = JaggedArrayBuilder.open(tmpdir, item_rank=2, dtype=jnp.float32)

            data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            builder.append(data)

            with pytest.raises(ValueError):
                builder[::2]
