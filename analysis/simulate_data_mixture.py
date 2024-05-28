import yaml
import numpy
from pathlib import Path

from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.shard_cache import ShardCache
from levanter.data.text import TokenSeqDataset, LMDatasetConfig


DATA_CONFIG = "config/data/dolma_olmo_paloma.yaml"
CACHE_DIR = "scratch/cache"



def construct_small_data_cache(
    path, num_shards=8, chunk_size=512, doc_len=128, vocab_size=1024
) -> tuple[LMDatasetConfig, dict[str, ShardCache]]:
    from levanter.data.shard_cache import SerialCacheWriter

    rng = numpy.random.default_rng(0)

    caches = {}

    for split in ["train", "validation"]:
        with SerialCacheWriter(f"{path}/cache/{split}", chunk_size) as writer:
            for shard in range(num_shards):
                writer.write_batch({"input_ids": rng.integers(0, vocab_size, size=(chunk_size, doc_len))})
        caches[split] = writer.result()

    config = LMDatasetConfig(
        train_urls=[f"file://{path}/train/docs.jsonl"],
        validation_urls=[f"file://{path}/validation/docs.jsonl"],
        cache_dir=f"{path}/cache",
        vocab_size=vocab_size,
        tokenizer="passthrough",
    )

    return config, caches


def simulate_olmo():
    seq_len = 10
    num_docs = 1000
    # load data config
    with open(DATA_CONFIG, "r") as f:
        data_config = yaml.safe_load(f)
    weights_config = data_config["train_weights"]

    # prepare data cache
    datasets = {}
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    for data_name in weights_config.keys():
        data_name = data_name.replace(" ", "_")
        construct_small_data_cache(
            f"{CACHE_DIR}/{data_name}", num_shards=1, chunk_size=num_docs, doc_len=seq_len
        )
        ds = TokenSeqDataset.load(seq_len, f"{CACHE_DIR}/{data_name}/cache/train")
        datasets[data_name] = ds
    
    # compare mixture with different strategies
    dataset = MixtureDataset(
        datasets=datasets,
        weights=weights_config,
        stop_strategy=StopStrategy.FIRST_STOP_STRATEGY,
    )
    for idx, content in enumerate(dataset):
        # print(f"idx: {idx}, content: {content}")
        if idx > 10000:
            break


if __name__ == "__main__":
    simulate_olmo()
