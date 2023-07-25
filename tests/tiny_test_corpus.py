import json
import os

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
