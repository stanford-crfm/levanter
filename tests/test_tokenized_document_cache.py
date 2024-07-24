import tempfile
from typing import List, Sequence, TypeVar

import pytest
import ray
from transformers import AutoTokenizer, BatchEncoding

from levanter.data.shard_cache import build_or_load_cache
from levanter.data.sharded_dataset import ShardedDataset, TextUrlDataset
from levanter.data.text import TokenizedDocumentCache
from levanter.utils.py_utils import logical_cpu_core_count
from test_utils import IdentityProcessor, ShardsDataset, SingleShardDocumentSource, skip_in_ci


tokenizer = AutoTokenizer.from_pretrained("gpt2")

T = TypeVar("T")


def setup_module(module):
    ray_designated_cores = max(1, logical_cpu_core_count())
    ray.init("local", num_cpus=ray_designated_cores)


def teardown_module(module):
    ray.shutdown()


@pytest.mark.ray
def test_index_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dataset = [""]
        source = SingleShardDocumentSource(empty_dataset)
        cache = TokenizedDocumentCache.build_or_load(
            f"{tmpdir}/cache",
            source,
            tokenizer,
            flatten_docs=True,
            enforce_bos=False,
            enforce_eos=False,
            override_resources={"num_cpus": 1},
        )

        for chunk in cache:
            assert chunk["input_ids"].size == 0


@pytest.mark.ray
def test_index_no_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dataset = []
        source = SingleShardDocumentSource(empty_dataset)
        cache = TokenizedDocumentCache.build_or_load(
            f"{tmpdir}/cache",
            source,
            tokenizer,
            flatten_docs=True,
            enforce_eos=False,
            override_resources={"num_cpus": 1},
        )

        for chunk in cache:
            pytest.fail("Should not have any chunks")


@skip_in_ci
@pytest.mark.ray
def test_doc_cache_reproduces_data_one_batch_per_shard():
    def doc_i(i: int):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1)))]))

    num_docs = 10
    docs = [doc_i(j) for j in range(num_docs)]

    class OneDocPerShardSource(ShardedDataset[T]):
        def __init__(self, docs: List[T]):
            self.docs = docs

        @property
        def shard_names(self) -> Sequence[str]:
            return [str(i) for i in range(len(self.docs))]

        def open_shard_at_row(self, shard_name: str, row: int):
            if row != 0:
                raise ValueError(f"Expected row 0, got {row}")

            return [self.docs[int(shard_name)]]

    source = OneDocPerShardSource(docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        build_or_load_cache(f"{tmpdir}/cache", source, IdentityProcessor(), await_finished=True)
        cache = TokenizedDocumentCache.load(f"{tmpdir}/cache", flatten_docs=False)

        result = list(cache)

        assert len(result) == num_docs
        # sort the docs by input_ids b/c the order is not guaranteed
        for i in range(len(result)):
            as_listed = BatchEncoding(data={k: [vv.tolist() for vv in v] for k, v in result[i].items()})
            assert as_listed == docs[i]


@skip_in_ci
@pytest.mark.ray
@pytest.mark.parametrize("batch_size", list([1, 2, 3, 8]))
def test_doc_cache_reproduces_data_multi_docs_per_batch_sharded(batch_size):
    def batch_docs(doc_ids):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1))) for i in doc_ids]))

    num_docs = 10
    batches = [batch_docs([j, j + 1]) for j in range(0, num_docs, batch_size)]

    source = ShardsDataset([[b] for b in batches])
    with tempfile.TemporaryDirectory() as tmpdir:
        build_or_load_cache(f"{tmpdir}/cache", source, IdentityProcessor())
        cache = TokenizedDocumentCache.load(f"{tmpdir}/cache", flatten_docs=True)

        result = list(cache)

        assert len(result) == len(batches)

        def list_in_list(a, b):
            """checks if a is a contiguous sublist of b"""
            n = len(a)
            return any((list(a) == list(b[i : i + n])) for i in range(len(b) - n + 1))

        # all we can really assert is that every doc from docs is in the result as a sublist
        for i in range(len(batches)):
            doc_tokens = batches[i]["input_ids"][0]
            found = False
            for j in range(len(result)):
                # check if the doc is in this result doc
                found = list_in_list(doc_tokens, result[j]["input_ids"][0])
                if found:
                    break
            assert found


@skip_in_ci
@pytest.mark.ray
def test_doc_cache_sharding():
    def doc_i(i: int):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1)))]))

    num_docs = 25
    num_shards = 12
    docs = [doc_i(j) for j in range(num_docs)]
    # group into num_shards groups
    doc_shards = [docs[i : i + num_docs // num_shards] for i in range(0, num_docs, num_docs // num_shards)]

    with tempfile.TemporaryDirectory() as tmpdir:
        source = ShardsDataset(doc_shards)
        build_or_load_cache(f"{tmpdir}/cache", source, IdentityProcessor())

        # must evenly divide num_shards
        num_shards_rebuild = [1, 2, 3, 4, 6, 12]

        for open_shards in num_shards_rebuild:
            cache = TokenizedDocumentCache.load(f"{tmpdir}/cache", flatten_docs=False)
            reconstructed = []

            for shard_idx in range(0, open_shards):
                # now we shard the cache
                c = cache.shard(shard_idx, open_shards)
                reconstructed.extend([d for b in c for d in _unbatch_encoding(b)])

            assert len(reconstructed) == num_docs

            # sort the docs by input_ids b/c the order is not guaranteed
            reconstructed.sort(key=lambda x: x["input_ids"][0][0])  # extra [0] for batchiness
            for i in range(len(reconstructed)):
                as_listed = BatchEncoding(data={k: [vv.tolist() for vv in v] for k, v in reconstructed[i].items()})
                assert as_listed == docs[i]


def _unbatch_encoding(enc: BatchEncoding):
    docs = []
    for i in range(len(enc["input_ids"])):
        docs.append(BatchEncoding(data={k: [v[i]] for k, v in enc.items()}))
    return docs


@pytest.mark.ray
def test_cache_fails_with_different_tokenizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(f"{tmpdir}/data.txt", "w") as f:
            f.write("")

        dataset = TextUrlDataset(
            [f"{tmpdir}/data.txt"],
        )

        tokenizer_a = AutoTokenizer.from_pretrained("microsoft/phi-2")
        tokenizer_b = AutoTokenizer.from_pretrained("google/flan-t5-small")

        TokenizedDocumentCache.build_or_load(
            tmpdir,
            dataset,
            tokenizer=tokenizer_a,
        )

        # Loading with the original tokenizer should be fine.
        TokenizedDocumentCache.build_or_load(
            tmpdir,
            dataset,
            tokenizer=tokenizer_a,
        )

        # Loading with a different tokenizer should error out.
        with pytest.raises(ValueError):
            TokenizedDocumentCache.build_or_load(
                tmpdir,
                dataset,
                tokenizer=tokenizer_b,
            )
