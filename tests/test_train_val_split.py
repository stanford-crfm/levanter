import jax
import pytest

from levanter.data import ListAsyncDataset
from levanter.data.text import LMDatasetConfig


@pytest.mark.asyncio
async def test_basic_split():
    """Test basic 80-20 split functionality"""
    # Create a simple dataset
    data = list(range(100))
    ds = ListAsyncDataset(data, is_complete=True)

    config = LMDatasetConfig(
        split_fraction=0.8,
        split_key=jax.random.PRNGKey(0),
    )

    # Mock the token_seq_dataset method to return our test dataset
    config.token_seq_dataset = lambda split, seq_len, monitors: ds

    # Get train and validation sets
    train_ds = config.train_set(seq_len=1, key=jax.random.PRNGKey(1))
    val_ds = config.validation_set(seq_len=1)

    # Check lengths
    train_indices = list(range(await train_ds.async_len()))
    val_indices = list(range(await val_ds.async_len()))

    train_len = len(await train_ds.get_batch(train_indices))
    val_len = len(await val_ds.get_batch(val_indices))

    assert train_len == 80
    assert val_len == 20
    assert train_len + val_len == len(data)


@pytest.mark.asyncio
async def test_disjoint_split():
    """Test that train and validation sets are disjoint"""
    data = list(range(100))
    ds = ListAsyncDataset(data, is_complete=True)

    config = LMDatasetConfig(
        split_fraction=0.8,
        split_key=jax.random.PRNGKey(0),
    )

    config.token_seq_dataset = lambda split, seq_len, monitors: ds

    train_ds = config.train_set(seq_len=1, key=jax.random.PRNGKey(1))
    val_ds = config.validation_set(seq_len=1)

    train_items = set(await train_ds.get_batch(list(range(await train_ds.async_len()))))
    val_items = set(await val_ds.get_batch(list(range(await val_ds.async_len()))))

    print(train_items)
    print(val_items)

    # Check sets are disjoint
    assert len(train_items.intersection(val_items)) == 0
    # Check union covers all data
    assert train_items.union(val_items) == set(data)


@pytest.mark.asyncio
async def test_deterministic_split():
    """Test that splits are deterministic with same key"""
    data = list(range(100))
    ds = ListAsyncDataset(data, is_complete=True)

    key = jax.random.PRNGKey(0)

    # Create two configs with same key
    config1 = LMDatasetConfig(split_fraction=0.8, split_key=key)
    config2 = LMDatasetConfig(split_fraction=0.8, split_key=key)

    config1.token_seq_dataset = lambda split, seq_len, monitors: ds
    config2.token_seq_dataset = lambda split, seq_len, monitors: ds

    # Get train sets from both configs
    train_ds1 = config1.train_set(seq_len=1, key=jax.random.PRNGKey(1))
    train_ds2 = config2.train_set(seq_len=1, key=jax.random.PRNGKey(1))

    train_items1 = await train_ds1.get_batch(list(range(await train_ds1.async_len())))
    train_items2 = await train_ds2.get_batch(list(range(await train_ds2.async_len())))

    assert train_items1 == train_items2


@pytest.mark.asyncio
async def test_different_keys_different_splits():
    """Test that different keys produce different splits"""
    data = list(range(100))
    ds = ListAsyncDataset(data, is_complete=True)

    config1 = LMDatasetConfig(split_fraction=0.8, split_key=jax.random.PRNGKey(0))
    config2 = LMDatasetConfig(split_fraction=0.8, split_key=jax.random.PRNGKey(1))

    config1.token_seq_dataset = lambda split, seq_len, monitors: ds
    config2.token_seq_dataset = lambda split, seq_len, monitors: ds

    train_ds1 = config1.train_set(seq_len=1, key=jax.random.PRNGKey(2))
    train_ds2 = config2.train_set(seq_len=1, key=jax.random.PRNGKey(2))

    train_items1 = await train_ds1.get_batch(list(range(await train_ds1.async_len())))
    train_items2 = await train_ds2.get_batch(list(range(await train_ds2.async_len())))

    assert train_items1 != train_items2


@pytest.mark.asyncio
async def test_edge_case_splits():
    """Test edge cases for split fractions"""
    data = list(range(100))
    ds = ListAsyncDataset(data, is_complete=True)

    # Test with very small split
    config = LMDatasetConfig(split_fraction=0.01, split_key=jax.random.PRNGKey(0))
    config.token_seq_dataset = lambda split, seq_len, monitors: ds

    train_ds = config.train_set(seq_len=1, key=jax.random.PRNGKey(1))
    val_ds = config.validation_set(seq_len=1)

    train_len = len(await train_ds.get_batch(list(range(await train_ds.async_len()))))
    val_len = len(await val_ds.get_batch(list(range(await val_ds.async_len()))))

    assert train_len == 1
    assert val_len == 99

    # Test with very large split
    config = LMDatasetConfig(split_fraction=0.99, split_key=jax.random.PRNGKey(0))
    config.token_seq_dataset = lambda split, seq_len, monitors: ds

    train_ds = config.train_set(seq_len=1, key=jax.random.PRNGKey(1))
    val_ds = config.validation_set(seq_len=1)

    train_len = len(await train_ds.get_batch(list(range(await train_ds.async_len()))))
    val_len = len(await val_ds.get_batch(list(range(await val_ds.async_len()))))

    assert train_len == 99
    assert val_len == 1


def test_invalid_split_fractions():
    """Test that invalid split fractions raise appropriate errors"""
    # Test split fraction = 0
    with pytest.raises(ValueError, match="split_fraction must be between 0 and 1"):
        LMDatasetConfig(split_fraction=0, split_key=jax.random.PRNGKey(0))

    # Test split fraction = 1
    with pytest.raises(ValueError, match="split_fraction must be between 0 and 1"):
        LMDatasetConfig(split_fraction=1, split_key=jax.random.PRNGKey(0))

    # Test negative split fraction
    with pytest.raises(ValueError, match="split_fraction must be between 0 and 1"):
        LMDatasetConfig(split_fraction=-0.1, split_key=jax.random.PRNGKey(0))

    # Test split fraction > 1
    with pytest.raises(ValueError, match="split_fraction must be between 0 and 1"):
        LMDatasetConfig(split_fraction=1.1, split_key=jax.random.PRNGKey(0))


def test_missing_split_key():
    """Test that missing split key raises appropriate error"""
    with pytest.raises(ValueError, match="split_key must be provided when split_fraction is set"):
        LMDatasetConfig(split_fraction=0.8, split_key=None)


@pytest.mark.asyncio
async def test_empty_dataset():
    """Test splitting an empty dataset"""
    data = []
    ds = ListAsyncDataset(data, is_complete=True)

    config = LMDatasetConfig(split_fraction=0.8, split_key=jax.random.PRNGKey(0))
    config.token_seq_dataset = lambda split, seq_len, monitors: ds

    train_ds = config.train_set(seq_len=1, key=jax.random.PRNGKey(1))
    val_ds = config.validation_set(seq_len=1)

    # Empty batch should raise ValueError
    with pytest.raises(ValueError, match="Dataset is empty"):
        await train_ds.get_batch([])
    with pytest.raises(ValueError, match="Dataset is empty"):
        await val_ds.get_batch([])
