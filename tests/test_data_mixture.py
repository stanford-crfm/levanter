import tempfile
import ray
from transformers import BatchEncoding

from levanter.data.shard_cache import build_cache
from levanter.data.text import MixtureDataset, TokenizedDocumentCache, FIRST_STOP_STRATEGY, ALL_STOP_STRATEGY
from levanter.utils.py_utils import logical_cpu_core_count
from test_utils import IdentityProcessor, SingleShardDocumentSource


def setup_module(module):
    ray_designated_cores = max(1, logical_cpu_core_count())
    ray.init("local", num_cpus=ray_designated_cores)


def teardown_module(module):
    ray.shutdown()


def test_mixture_dataset():
    seq_len = 10

    def doc_i(i: int):
        return BatchEncoding(data=dict(input_ids=[list(range(seq_len * i, seq_len * (i + 1)))]))

    num_docs_1, num_docs_2 = 10, 20
    docs_1 = [doc_i(j) for j in range(num_docs_1)]
    docs_2 = [doc_i(j) for j in range(num_docs_2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        source_1 = SingleShardDocumentSource(docs_1)
        build_cache(f"{tmpdir}/cache_1", source_1, IdentityProcessor())
        cache_1 = TokenizedDocumentCache.load(f"{tmpdir}/cache_1", flatten_docs=False)

        source_2 = SingleShardDocumentSource(docs_2)
        build_cache(f"{tmpdir}/cache_2", source_2, IdentityProcessor())
        cache_2 = TokenizedDocumentCache.load(f"{tmpdir}/cache_2", flatten_docs=False)

        mixture_data_config = {
            "doc_caches": {"1": cache_1, "2": cache_2},
            "seq_len": seq_len,
        }
        mixture_1_only = MixtureDataset(
            weights={"1": 1.0, "2": 0.0},
            stop_strategy=FIRST_STOP_STRATEGY,
            **mixture_data_config,
        )
        counter = 0
        for batch in mixture_1_only:
            assert batch.shape == (seq_len,)
            counter += 1
        assert counter == 10

        mixture_balanced_first = MixtureDataset(
            weights={"1": 0.5, "2": 0.5},
            stop_strategy=FIRST_STOP_STRATEGY,
            **mixture_data_config,
        )
        counter_first = sum([1 for _ in mixture_balanced_first])

        mixture_balanced_all = MixtureDataset(
            weights={"1": 0.5, "2": 0.5},
            stop_strategy=ALL_STOP_STRATEGY,
            **mixture_data_config,
        )
        counter_all = sum([1 for _ in mixture_balanced_all])
        assert counter_first < counter_all
