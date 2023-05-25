# ---
layout: blog
title: Levanter — Towards Scaling Reproducible Foundational Models with Jax
authors:
    - name: David Hall
      url: TODO
    - name: Percy Liang
      url: https://cs.stanford.edu/~pliang/
display: False
---
> <div class="blog-tagline">
    <strong> We introduce our projects
        <a href="https://github.com/stanford-mercury/levanter" target="_blank">Haliax and Levanter</a>, our code
        and infrastructure for training reproducible, legible foundation models models using Jax. We also XXX checkpoints.
    </strong>
> </div>


## Introduction

The growth of artificial intelligence and machine learning has brought about the need for scalable and reproducible models.
To address this, at the [Center for Research on Foundation Models](https://crfm.stanford.edu), we have created two new tools — [Levanter and Haliax](https://github.com/stanford-crfm/levanter).
They form a new code base for training foundational models with the promise of flexibility, modularity, efficiency,
and scale, as well as strong guarantees about reproducibility.

Levanter is a [Jax](https://github.com/google/jax)-based codebase for training foundation models that is designed to be
flexible, modular, and accessible, while still being performant and scalable. In particular, Levanter is designed to be
able to train models on a variety of different hardware, including GPUs, TPUs, and TPU pods. Levanter also offers strong
guarantees about reproducibility, and we have released checkpoints for a number of models trained with Levanter.
Haliax is a named tensor library that we developed to make it easier to write legible, composable code while
still maintaining efficiency and scalability.

Today, we're releasing the first version of Levanter and Haliax, along with several models trained with Levanter. We
hope that these libraries will be useful to the community, and we look forward to seeing what people do with them.

## The Landscape of Foundation Model Training Frameworks

Numerous foundation model training frameworks exist in the community, each with its strengths and focus.
For large language models (LLMs) (the focus of this release),
the most well-known in the open source community is probably NVIDIA's PyTorch-based [Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
and its many derivatives, including EleutherAI's [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) codebase. MosaicML
has recently released [LLM Foundry](https://github.com/mosaicml/llm-foundry), which we used to train
[BioMedLM](https://crfm.stanford.edu/2022/12/15/biomedlm.html) last autumn.

In the Jax community, there are a number of libraries popping up. Google has released [T5X](https://github.com/google-research/t5x)
and [MaxText](https://github.com/google/maxtext). There are also a number of independent libraries, including [EasyLM](https://github.com/young-geng/EasyLM)
and [JaxSeq](https://github.com/Sea-Snell/JAXSeq), both of which are based on [Flax](https://github.com/google/flax/)
and modified libraries from [Hugging Face Transformers](https://github.com/huggingface/transformers/). Salesforce has
released the Haiku-based [Jaxformer](https://github.com/salesforce/jaxformer).

Despite the wide array of existing frameworks, when we started, we found that none of them fully addressed our needs.
At CRFM, we wanted to achieve four goals:

* **Legibility**: We wanted to be able to write code that was easy to read and understand, and that could be composed easily.
* **Reproducibility**: We wanted to be able to reproduce our results exactly, even in the presence of preemption and
  restarting from checkpoints.
* **Scalability**: We wanted to be able to fully utilize the Cloud TPU resources we were given as part of Google's TPU Research Cloud program, as well as our own GPU resources.
* **Efficiency**: We wanted to achieve the other three goals without sacrificing (much) efficiency.

We chose [Jax](https://github.com/google/jax/) as our framework because it is a powerful, flexible, and performant,
and offers strong reproducibility guarantees. Jax works well on TPUs, while we found that PyTorch support was still uneven.
Jax is also a natural choice because it allows you to focus on the "what" of your code, and not on the "how:" details of
partitioning and communication can be left to the XLA compiler. Finally, Jax makes reproducibility easy, since it uses
bitwise deterministic PRNGs by default, with careful control over the PRNG state.

However, Jax is a low-level framework, and we found that, by itself, it did not provide the legibility that we wanted.

For us, legibility is a top concern. We therefore created two new libraries: **Haliax** and **Levanter**. Haliax is a
named tensor library that focuses on improving the legibility and compositionality of deep learning code while still
maintaining efficiency and scalability. Levanter is a library for training foundation models built on top of Haliax. In
addition to the goals of legibility, efficiency, and scalability, Levanter further strives for bitwise reproducibility,
meaning that the same code with the same data will produce the exact same result, even in the presence of preemption and
restarts.

## Haliax: Legibility via Named Tensors

Haliax is a Jax library for building neural networks with named tensors, built on Jax and [Equinox](https://github.com/patrick-kidger/equinox),
which is a neural network library built on Jax that provides a familiar, PyTorch-like module structure.

Named Tensors are a powerful abstraction that allow you to give names to the axes of your tensors. These names help
make your code more legible, more composable, and avoid bugs. In Haliax, they also form the basis of how we handle
scale with [Fully-Sharded Data Parallel](https://engineering.fb.com/2021/07/15/open-source/fsdp/). (See below for our tutorial)

Haliax is modeled on Alexander Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor). In particular, he argues that:

* Named axes are more semantically meaningful and rely less on bitrot-prone comments.
* Broadcasting leads to unreadable strings of `view`s and `squeeze`s. (To this, I'd add implicit broadcasting is a source of bugs.)
* Named axes allow you to abstract over unreferenced dimensions, making code more flexible for things like multi-headed attention or flexible attention masking.

Let's take a look at a practical example of how Haliax can be used.
Here's a minimal attention module implementation in Haliax. For a more detailed introduction,
please see the [Haliax tutorial](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import haliax as hax
import haliax.nn as hnn

# Named Axes for Tensor Dimensions
Pos = hax.Axis("position", 1024)  # sequence
KPos = Pos.alias("key_position")  # key sequence for attention
Head = hax.Axis("head", 8)  # number of attention heads
Key = hax.Axis("key", 64)  # key size
Embed = hax.Axis("embed", 512)  # embedding size

# Despite making no reference to batching or heads, this same implementation is also batch-capable and multi-headed
# (or multi-query) and even supports attending to or from non-sequential keys (e.g. image patches)
def attention(Key, KPos, query, key, value, mask):
  # how similar is each query to each key
  scores = hax.dot(Key, query, key) / jnp.sqrt(Key.size)

  # mask out invalid positions
  if mask is not None:
    scores -= 1E9 * (1.0 - mask)

  # convert to probabilities
  scores = hax.nn.softmax(scores, KPos)

  # weighted sum of values
  result = hax.dot(KPos, scores, value)

  return result


# Causal Mask means that if pos >= key_pos, then pos can attend to key_pos
causal_mask = hax.arange(Pos).broadcast_axis(KPos) >= hax.arange(KPos)

class Attention(eqx.Module):
  proj_qkv: hnn.Linear  # input projection from [Embed] -> [(q, k, v), Head, Key]
  proj_answer: hnn.Linear  # output projection from [Head, Key] -> [Embed]

  @staticmethod
  def init(Embed, Head, Key, *, key: PRNGKey):
    Qkv = hax.Axis("qkv", 3)  # create all three at once

    k_qkv, k_ans = jax.random.split(key, 2)
    proj_qkv = hnn.Linear.init(In=Embed, Out=(Qkv, Head, Key), key=k_qkv)
    proj_answer = hnn.Linear.init(In=(Head, Key), Out=Embed, key=k_ans)
    return Attention(proj_qkv, proj_answer)

  def __call__(self, x, mask):
    qkv_out = self.proj_qkv(x)
    q, k, v = qkv_out.unbind("qkv")

    # Rename k and v's Pos as Haliax doesn't support unnamed axes or duplicate axes
    k = k.rename({Pos: KPos})
    v = v.rename({Pos: KPos})

    answers = attention(Key, KPos, q, k, v, causal_mask)

    x = self.proj_answer(answers)
    return x
```

We use named axes both to improve legibility and to enable scale: named axes are the basis of our
[Fully-Sharded Data Parallel](https://engineering.fb.com/2021/07/15/open-source/fsdp/) implementation. FSDP can
be added to the training code with about 10 lines of code, enabling scale to at least 256 TPU cores (which is
as many as we can get our hands on) and at least 65b parameters (which is way bigger than we have compute for).

### Haliax Tutorials

For more details, please see our interactive tutorials on Colab:

* [Introduction to Haliax with Transformers](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC?usp=sharing)
* [Scaling Transformers in Haliax](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz?usp=sharing), including FSDP in Jax.

## Levanter: Bitwise Determinism with Jax

Levanter is a library for training foundation models built on top of Haliax. It provides a complete pipeline
for training a GPT-2-like Transformer, complete with data preparation, logging, training, checkpointing, evaluation, and export,
while maintaining bitwise reproducibility throughout.

We have used Levanter to train models as large as 6.7b parameters on a v3-256, and have run experiments showing that it can scale
up to least 65b parameters.

### Bitwise Reproducibility

One of the benefits of Jax is that it offers strong guarantees for reproducibility. In particular, Jax's fine-grained
control over random number generation makes it easy to ensure bitwise reproducibility, especially when using TPUs.
Levanter takes advantage of this to offer bitwise reproducibility for training runs, even after preemption. As an example,
here is a screenshot of a training run being resumed multiple times, even on different TPU pod slices:

![plot showing bitwise reproducibility with four training runs superimposed with the exact same loss curve](figures/bitwise_repro_curve.png)

The fact that you can't make out the different lines is the point: the training runs are bitwise identical,
a huge advantage for debugging and reproducibility.

### Efficiency and Scale

XXX something something v3-256 scaling numbers?

### Other Features

* **Preprocessing**: Levanter uses [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers/) to preprocess text
using distributed preprocessing backed by [Ray](https://www.ray.io/))
* **Training**: In addition to Jax and Haliax, Levanter uses [Optax](https://github.com/deepmind/optax) for optimization,
  (though our new optimizer, [Sofia](https://arxiv.org/abs/2305.14342), is coming to Levanter soon!)
* **Logging**: Logging is done with [WandB](https://wandb.ai/), complete with a fancy online visualization of the validation set during training.
* **Checkpointing**: Distributed checkpointing is supported via Google's [TensorStore](https://google.github.io/tensorstore/) library.
* **Export**: We also support exporting models to the Hugging Face Hub, with export compatibler with Pytorch and Transformers via [SafeTensors](https://github.com/huggingface/safetensors).
* **Stability**: The GPT-2 implementation uses the [Mistral stability trick](https://crfm.stanford.edu/2021/08/26/mistral.html) to improve stability during training.


#### Live Visualization during Training

While collaborating with teams to build domain-specific models, we've found that data preparation can be a significant challenge.
As an example, it can be difficult to identify issues in the data that are causing
lower than expected loss. As a small step towards addressing this, we have added a feature to Levanter that allows you
to visualize the probability of each token in a sample of the validation set during training. For novel data sets,
this has allowed us to identify, e.g., highly but not perfectly redundant data (what we call "madlib duplicates") or other
issues that might be causing abnormally low (or high) loss. We've also used it to qualitatively assess how alternative architectures
learn differently from Transformers.

Here is an example of the token probability visualization in action on a small, quick training run:

![video showing heat map of token probabilities for a sample of the validation set evolving as training progresses](figures/token_probabilities.mov)

The darker, more purple the color, the lower the probability of the token. The lighter, more yellow the color, the higher the probability.
This visualization is logged to WandB as training progresses and can be viewed interactively. I have found this to be a
nice alternative to just staring obsessively at the loss curve, which is what I usually do.


#### On-Demand Preprocessing

We've found that in many cases, teams would benefit from the ability to preprocess data on the fly, and Levanter
provides this capability. Levanter can automatically spin up a Ray cluster using the nodes being used for training,
using the typically impressive CPUs of those machines to preprocess data. This is especially useful for large data sets
like [The Pile](https://pile.eleuther.ai/) or the [Red Pajama](https://github.com/togethercomputer/RedPajama-Data) dataset.
Preprocessing can also be performed offline using a Ray cluster, or on a single machine. In all cases, the caches
produced by preprocessing are fully reproducible, so that we can assure bitwise reproducibility even when preprocessing
is performed on different machines.

Soon, we will enable training while tokenization is still in progress, which will allow you to start training
on a data set before it is fully tokenized. This will be especially useful when iterating on formatting for large data sets,
where preprocessing can take a long time. We also aim to make resuming from preemption even faster than it is now.

### Getting Started with Levanter

XXX

### Released Models

Along with the release of the code, we are releasing a few models trained using Levanter. These models are available on
the [HF model hub](XXX) and can be used with the Hugging Face Transformers library. We have more in development and will
be releasing them as they become available.

XXX TODO: verify this is what we're releasing, add links
- GPT-2-1536 (1.45B parameters) XXX, including checkpoints every 10,000 steps
- XXX music model (link to other blog post XXX)
- Backpacks? XXX

## Future and Conclusion

This is just the beginning for Levanter. In the future, look for:
* more models on interesting problem domains,
* scaled up versions of new architectures developed here at Stanford and elsewhere,
* new training techniques, including the newly released [Sofia](https://arxiv.org/abs/2305.14342) optimizer,
* and larger models

Levanter is still a work in progress, but we are excited to share it with the community. We hope that Levanter will be
useful to others who are interested in training foundation models using Jax and TPUs. Please join us on our journey! You
can find us on [GitHub](https://github.com/stanford-crfm/levanter), [Twitter](https://twitter.com/StanfordCRFM), or on
the (unofficial) [Jax LLM Discord](https://discord.gg/CKazXcbbBm). (And by the way, [we're hiring](https://crfm.stanford.edu/apply.html)!)

## Acknowledgements

In addition to the generous support of the Google TPU Research Cloud, we would like to thank the following people for their help and support:

* John Thickstun, Sidi Lu, John Hewitt, and others for being early adopters and providing feedback. We really appreciate your patience, support, and feedback.
* Yifan Mai, Tony Lee, Jason Bolton, Ivan Zhou, and the rest of the CRFM engineering team for support and discussions.
* The TRC team for support getting spun up on TPUs and for making the Cloud TPUs available to us.
* Roy Frostig, Sholto Douglas, Skye Wanderman-Miln, and the rest of the Jax team for help with debugging and support.
* Sidd Karamcheti for support and conversations
