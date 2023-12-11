import tempfile

from levanter.data.text import LMDatasetConfig


def test_dont_blow_up_without_validation_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LMDatasetConfig(
            train_urls=["kaa"],
            validation_urls=[],
            cache_dir=tmpdir,
        )

    # mostly just making sure this doesn't blow up
    assert config.validation_set(10) is None
