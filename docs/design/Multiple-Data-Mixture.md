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

The common approach at training, used in LLaMA, MPT, and Falcon models, is to sample from each domain in 
the dataset uniformly, based on the percentage of tokens in each domain. Recently, we have seen a new line 
of work that uses a weighted sampling approach, such as [PaLM](https://arxiv.org/abs/2204.02311) and 
[DoReMi](https://arxiv.org/abs/2305.10429). 

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

### Mixing at Batch Level
Currently at training, `ShardedBatchLoader` takes in a single dataset of TokenSeqDataset class, 
and implements the batching and sharding logic. Therefore, the mixing of multiple datasets should 
be implemented at ShardedBatchLoader. 

Specifically, at every step of `ShardedBatchLoader.__iter__()`, instead of slicing B sequences from 
a single dataset, we will sample B sequences from N datasets, where N is the number of datasets.

*TODO: remove the following question*
> Question: I don't following understand the logic that, at `ShardedBatchLoader.__iter__()`, we have
a for loop over `one_item_generator`, then at every step, we slice from `one_item_generator` to get 
B sequence, what do we want to achieve with the first for loop?

### Weighted Sampling
At every step of `ShardedBatchLoader.__iter__()`, we need to sample B sequences from N datasets. 
The weights of the datasets are used to sample from N datasets. Specifically, we will sample
B sequences from the i-th dataset with probability `weight[i] / sum(weight)`. 

### Deterministic Batches
I am not sure if we can still achieve deterministic batches with weighted sampling.

### Validation Set
We will apply the same logic to `ReplicatedBatchLoader` for validation sets. 

*TODO: remove the following question*
> Question: right now does Levanter supports a dataset that only has a training set but no validation set?
