import tempfile

import pytest
from transformers import AutoTokenizer, BatchEncoding

from levanter.data.text import TokenizedDocumentCache


tokenizer = AutoTokenizer.from_pretrained("gpt2")


def test_index_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dataset = [{"text": ""}]
        tokenized = [tokenizer(x["text"]) for x in empty_dataset]
        cache = TokenizedDocumentCache.build_or_load(tokenized, f"{tmpdir}/cache", num_shards=1, flatten_docs=False)

        for chunk in cache:
            pytest.fail(f"Should not have any entries, but got {chunk}")


def test_doc_cache_reproduces_data_one_batch_per_file():
    def doc_i(i: int):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1)))]))

    for i in range(10):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizedDocumentCache.build_or_load(
                [doc_i(j) for j in range(i)], f"{tmpdir}/cache", num_shards=1, flatten_docs=False
            )

            result = list(cache)

            assert len(result) == i
            for j in range(len(result)):
                as_listed = BatchEncoding(data={k: [vv.tolist() for vv in v] for k, v in result[j].items()})
                assert as_listed == doc_i(j)


def test_doc_cache_reproduces_data_one_batch_per_file_sharded():
    def doc_i(i: int):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1)))]))

    num_docs = 10
    docs = [doc_i(j) for j in range(num_docs)]
    for num_shards in range(1, 10):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TokenizedDocumentCache.build_or_load(
                iter(docs), f"{tmpdir}/cache", num_shards=num_shards, flatten_docs=False
            )

            result = list(cache)

            assert len(result) == num_docs
            # sort the docs by input_ids b/c the order is not guaranteed
            result.sort(key=lambda x: x["input_ids"][0][0])  # extra [0] for batchiness
            for i in range(len(result)):
                as_listed = BatchEncoding(data={k: [vv.tolist() for vv in v] for k, v in result[i].items()})
                assert as_listed == docs[i]


def test_doc_cache_reproduces_data_multi_docs_per_batch_sharded():
    def batch_docs(doc_ids):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1))) for i in doc_ids]))

    num_docs = 10
    for batch_size in range(1, 10):
        batches = [batch_docs([j, j + 1]) for j in range(0, num_docs, batch_size)]
        for num_shards in range(1, 10):
            with tempfile.TemporaryDirectory() as tmpdir:
                cache = TokenizedDocumentCache.build_or_load(
                    iter(batches), f"{tmpdir}/cache", num_shards=num_shards, flatten_docs=True  # NB: flatten_docs=True
                )

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


def test_doc_cache_sharding():
    def doc_i(i: int):
        return BatchEncoding(data=dict(input_ids=[list(range(10 * i, 10 * (i + 1)))]))

    num_docs = 25
    num_shards = 12
    docs = [doc_i(j) for j in range(num_docs)]

    with tempfile.TemporaryDirectory() as tmpdir:
        TokenizedDocumentCache.build_or_load(
            iter(docs),
            f"{tmpdir}/cache",
            num_shards=num_shards,
            flatten_docs=False,
        )

        # must evenly divide num_shards
        num_shards_rebuild = [1, 2, 3, 4, 6, 12]

        for open_shards in num_shards_rebuild:
            cache = TokenizedDocumentCache.load(f"{tmpdir}/cache", flatten_docs=False)
            reconstructed = []

            for shard_idx in range(0, open_shards):
                # now we shard the cache
                c = cache.shard(shard_idx, open_shards)
                reconstructed.extend(list(c))

            assert len(reconstructed) == num_docs

            # sort the docs by input_ids b/c the order is not guaranteed
            reconstructed.sort(key=lambda x: x["input_ids"][0][0])  # extra [0] for batchiness
            for i in range(len(reconstructed)):
                as_listed = BatchEncoding(data={k: [vv.tolist() for vv in v] for k, v in reconstructed[i].items()})
                assert as_listed == docs[i]
