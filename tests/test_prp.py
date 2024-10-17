import jax.numpy as jnp
import jax.random as jrandom
import pytest

from levanter.data._prp import Permutation


def test_permutation_creates_valid_instance():
    length = 100
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    assert permutation.length == length
    assert permutation._a > 0 and permutation._a < length
    assert permutation._b >= 0 and permutation._b < length


def test_permutation_with_single_index_returns_correct_value():
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    index = 5
    result = permutation(index)
    assert isinstance(result, int)
    assert result != index  # In most cases, result should not equal the input for a permutation


def test_permutation_with_array_returns_correct_values():
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    indices = jnp.arange(length)
    results = permutation(indices)
    assert isinstance(results, jnp.ndarray)
    assert len(results) == length
    assert jnp.sum(results == indices) <= 2


def test_permutation_is_bijective_over_full_range():
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    indices = jnp.arange(length)
    permuted = permutation(indices)
    # Check if all elements are unique, which is a necessary condition for a bijective function
    assert len(jnp.unique(permuted)) == length


def test_permutation_handles_edge_case_length_one():
    length = 1
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    result = permutation(0)
    assert result == 0  # With length 1, the only valid output is the input it


def test_permutation_rejects_invalid_indices():
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    with pytest.raises(IndexError):
        permutation(-1)  # Test negative index
    with pytest.raises(IndexError):
        permutation(length)  # Test index equal to length


def test_permutation_is_deterministic():
    length = 4
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    indices = jnp.arange(length)
    results = permutation(indices)
    prng_key = jrandom.PRNGKey(0)
    permutation = Permutation(length, prng_key)
    results2 = permutation(indices)
    assert jnp.all(results == results2)


def test_permutation_is_deterministic1():
    length = 4
    prng_key = jrandom.PRNGKey(1)
    permutation = Permutation(length, prng_key)
    indices = jnp.arange(length)
    results = permutation(indices)
    prng_key = jrandom.PRNGKey(1)
    permutation = Permutation(length, prng_key)
    results2 = permutation(indices)
    assert jnp.all(results == results2)
