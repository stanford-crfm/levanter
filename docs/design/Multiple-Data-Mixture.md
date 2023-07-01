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

We still want to preserve the reproducibility and deterministic batches of Levaner, as specified in the 
[Data Loader design](Data-Loader-Design.md).

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

In the new design, users can specify multiple datasets in the form of a list and the weight of each
dataset, for example:

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

### Data Loader
The data loader will be modified to support the new configuration. Specifically, the data loader will
take in a list of datasets, each with a weight. The data loader will then sample the number of tokens
from each dataset based on the weight.

### Deterministic Batches
The deterministic batches will be preserved. Specifically, the data loader will sample the number of
tokens from each dataset based on the weight, and then sample the tokens from each dataset uniformly.
