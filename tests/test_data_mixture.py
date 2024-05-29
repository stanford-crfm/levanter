import tempfile

import tiny_test_corpus
from levanter.data import Dataset
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.text import TokenSeqDataset


class ListDataset(Dataset[list]):
    def __init__(self, data: list):
        self.data = data

    def __iter__(self):
        return iter(self.data)


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
            key=0,
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
            key=0,
        )
        counter_first = sum([1 for _ in mixture_balanced_first])

        mixture_balanced_all = MixtureDataset(
            datasets=datasets,
            weights={"1": 0.5, "2": 0.5},
            stop_strategy=StopStrategy.ALL_STOP_STRATEGY,
            key=0,
        )
        counter_all = sum([1 for _ in mixture_balanced_all])
        assert counter_first < counter_all

        # test normalized weights
        mixture_normalized = MixtureDataset(
            datasets=datasets,
            weights={"1": 2.0, "2": 2.0},
            stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
            key=0,
        )
        assert mixture_normalized.weights["1"] == mixture_normalized.weights["2"] == 0.5


def test_restart_strategy_gets_the_right_average():

    num_docs_1, num_docs_2 = 10, 20
    ds1 = ListDataset([1 for _ in range(num_docs_1)])
    ds2 = ListDataset([2 for _ in range(num_docs_2)])

    datasets = {"1": ds1, "2": ds2}
    mixture_balanced_restart = MixtureDataset(
        datasets=datasets,  # type: ignore
        weights={"1": 0.6, "2": 0.4},
        stop_strategy=StopStrategy.RESTART_STRATEGY,
        key=0,
    )

    # ensure we get the right long run average
    NUM_SAMPLES = 2300

    # variance of a bernoulli distribution is p(1-p) ≈ 0.24
    # to get a 95% confidence interval of 0.02, we need ~2300 samples

    # we expect to get roughly 60% 1s and 40% 2s
    num_ones = 0
    for i, ex in enumerate(mixture_balanced_restart):
        if ex == 1:
            num_ones += 1
        if i >= NUM_SAMPLES:
            break

    assert 0.58 < num_ones / NUM_SAMPLES < 0.62

    # now just to verify, stop_first won't give us the same average

    num_total = 0
    num_ones = 0

    mixture_balanced_first = MixtureDataset(
        datasets=datasets,  # type: ignore
        weights={"1": 0.6, "2": 0.4},
        stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
        key=0,
    )

    for i, ex in enumerate(mixture_balanced_first):
        if ex == 1:
            num_ones += 1
        num_total += 1

    assert num_total < 30
    assert num_ones == num_docs_1
    assert num_ones / num_total < 0.55 or num_ones / num_total > 0.65
