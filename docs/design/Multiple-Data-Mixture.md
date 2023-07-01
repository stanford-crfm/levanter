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
2. At every batch, the weight of each domain is used to sample the number of tokens from the corresponding domain.

We still want to preserve the reproducibility and deterministic batches of Levaner, as specified in the 
[Data Loader design](Data-Loader-Design.md).
