# Weighted Data Mixture Design
> Design as of 2023-07

## Goals
Datasets for training language models are commonly sampled from a mixture of multiple domains or sources.
For example:
- The [Pile dataset](https://pile.eleuther.ai/) is a mixture of 22 different datasets, including
web data (24%), Wikipedia (9%), GitHub (4%), and more;
- The [RedPajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) is
a mixture of 1.2 tokens, from domains such as CommonCrawl (73%), GitHub (5%), Books (2.4%),
Wikipedia (2.4%), etc.
- The [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b) model is trained on 1T tokens from
sources including RedfinedWeb English (75%), Books (6%), Conversations (5%), etc.

The common approach at training, used in LLaMA, MPT, and Falcon models, is to sample token sequence from
the entire training corpus uniformly. GPT-3 and Pile uses heuristically-chosen domain weights.
Recently, we have seen a new line of work that applies systematic approach at determining optimal domain
weights, such as [PaLM](https://arxiv.org/abs/2204.02311) and [DoReMi](https://arxiv.org/abs/2305.10429).

In this design, we want to implement the weighted sampling approach in Levanter. Specifically:
1. Users can specify the weight of each domain in the dataset, through the configuration file.
2. At every batch, the weight of each domain is used to sample the number of tokens from the
corresponding domain.

Ideally, we still want to preserve the reproducibility and deterministic batches of Levanter,
as specified in the [Data Loader design](Data-Loader-Design.md).


## Design and Implementation
### Configuration
Currently, in a configuration file, under "data" section, users would only specify a single dataset,
for example:

```yaml
data:
  train_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
  validation_urls:
      - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
  cache_dir: "gs://pubmed-mosaic/tokenized/openwebtext/"
```

In the new design, users can specify multiple datasets as a list of LMDatasetConfig. Within
each LMDatasetConfig, users can specify the weight of each dataset, for example:

```yaml
data:
  datasets:
    - name: "openwebtext"
      weight: 0.5
      train_urls:
        - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
      validation_urls:
          - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
      cache_dir: "gs://pubmed-mosaic/tokenized/openwebtext/"
    - name: "reddit"
      weight: 0.5
      train_urls:
        - "gs://pubmed-mosaic/reddit-sharded/reddit_train.{1..128}-of-128.jsonl.gz"
      validation_urls:
        - "gs://pubmed-mosaic/reddit-sharded/reddit_val.{1..8}-of-8.jsonl.gz"
      cache_dir: "gs://pubmed-mosaic/tokenized/reddit/"
```

Note that this design is backward compatible, as users can still specify a single dataset in the
configuration file, and the weight of the dataset will be 1.0 by default.

### LMMixtureDatasetConfig
We will introduce a new data class `MixtureDatasetConfig`, which takes in a list of `LMDatasetConfig`
with weights. `MixtureDatasetConfig` provides a consistent interface as `LMDatasetConfig`, so that we don't need to introduce many special logic at training/evaluation for handling mixture datasets.


### MixtureDataset
Currently at training, `ShardedBatchLoader` takes in a `TokenSeqDataset` instance. `TokenSeqDataset`
iterates over documents and yield token sequences; `ShardedBatchLoader` implements the batching
and sharding logic.

To support a mixture of multiple datasets, we will implement a new `MixtureDataset` class, which
takes a list of datasets and yield token sequences, so that the implementation of `ShardedBatchLoader`
remains the same.

Specifically, in `MixtureDataset`, it takes in a list of datasets, each with its tokenized document
cache (of the `TokenizedDocumentCache` class) and weights (float). At every step of
`MixtureDataset.__iter__()`, it will sample a dataset from the list of datasets, with probability
proportional to the weight of the dataset. Then, it will yield a token sequence from the sampled
dataset.

### Validation Set
We will not apply weighted sampling to the validation set. Instead, we will report performance on each
validation set separately.
