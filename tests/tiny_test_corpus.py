import json
import os

import numpy

from levanter.data.audio import AudioIODatasetConfig
from levanter.data.shard_cache import ShardCache
from levanter.data.text import LMDatasetConfig


def _write_tiny_corpus(path):
    os.makedirs(f"{path}/train", exist_ok=True)
    with open(f"{path}/train/docs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"hello world {i} " * 100}))
            f.write("\n")

    os.makedirs(f"{path}/validation", exist_ok=True)
    with open(f"{path}/validation/docs.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"text": f"bye world {i} " * 100}))
            f.write("\n")


def tiny_corpus_config(path):
    _write_tiny_corpus(path)
    return LMDatasetConfig(
        train_urls=[f"file://{path}/train/docs.jsonl"],
        validation_urls=[f"file://{path}/validation/docs.jsonl"],
        cache_dir=f"{path}/cache",
    )


def tiny_asr_corpus_config(path):
    return AudioIODatasetConfig(
        id="WillHeld/test_librispeech_parquet",
        train_split="validation",
        validation_split="validation",
        cache_dir=f"{path}/cache_asr",
    )


def construct_small_data_cache(
    path, num_shards=8, chunk_size=512, doc_len=128, vocab_size=1024
) -> tuple[LMDatasetConfig, ShardCache]:
    from levanter.data.shard_cache import SerialCacheWriter

    rng = numpy.random.default_rng(0)

    for split in ["train", "validation"]:
        with SerialCacheWriter(f"{path}/cache/{split}", chunk_size) as writer:
            for shard in range(num_shards):
                writer.write_batch({"input_ids": rng.integers(0, vocab_size, size=(chunk_size, doc_len))})

    config = LMDatasetConfig(
        train_urls=[f"file://{path}/train/docs.jsonl"],
        validation_urls=[f"file://{path}/validation/docs.jsonl"],
        cache_dir=f"{path}/cache",
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    return config, writer.result()
