import tempfile

import tiny_test_corpus
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.text import TokenSeqDataset


def test_stop_strategies():
    seq_len = 10

    num_docs_1, num_docs_2 = 10, 20
    with tempfile.TemporaryDirectory() as tmpdir:
        # source_1 = SingleShardDocumentSource(docs_1)
        data_config, _ = tiny_test_corpus.construct_small_data_cache(
            f"{tmpdir}/cache_1", num_shards=1, chunk_size=num_docs_1, doc_len=seq_len
        )

        data_config, _ = tiny_test_corpus.construct_small_data_cache(
            f"{tmpdir}/cache_2", num_shards=1, chunk_size=num_docs_2, doc_len=seq_len
        )

        ds1 = TokenSeqDataset.load(seq_len, f"{tmpdir}/cache_1/cache/train")
        ds2 = TokenSeqDataset.load(seq_len, f"{tmpdir}/cache_2/cache/train")

        # set reuseable config
        datasets = {"1": ds1, "2": ds2}
        # test mixture with all weights on one dataset
        mixture_1_only = MixtureDataset(
            datasets=datasets,
            weights={"1": 1.0, "2": 0.0},
            stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
        )
        counter = 0
        for batch in mixture_1_only:
            assert batch.shape == (seq_len,)
            counter += 1
        assert counter == 10

        # compare mixture with different strategies
        mixture_balanced_first = MixtureDataset(
            datasets=datasets,
            weights={"1": 0.5, "2": 0.5},
            stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
        )
        counter_first = sum([1 for _ in mixture_balanced_first])

        mixture_balanced_all = MixtureDataset(
            datasets=datasets,
            weights={"1": 0.5, "2": 0.5},
            stop_strategy=StopStrategy.ALL_STOP_STRATEGY,
        )
        counter_all = sum([1 for _ in mixture_balanced_all])
        assert counter_first < counter_all

        # test normalized weights
        mixture_normalized = MixtureDataset(
            datasets=datasets,
            weights={"1": 2.0, "2": 2.0},
            stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
        )
        assert mixture_normalized.weights["1"] == mixture_normalized.weights["2"] == 0.5
