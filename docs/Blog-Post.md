# ---
layout: blog
title: Levanter —  Scalable, Reproducible, Legible Foundation Models with Jax
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
They form a new code base for training foundatiol models with the promise of flexibility, modularity, efficiency,
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

Numerous foundation model training frameworks exist in the community, each with its strengths and focuses.
For large language models (LLMs) (the focus of this release),
the most well-known in the open source community is probably NVIDIA's PyTorch-based [Megatron-LM](https://github.com/NVIDIA/Megatron-LM),
and its many derivatives, including EleutherAI's [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) codebase.
Meta has [MetaSeq](https://github.com/facebookresearch/metaseq) as well as [FairScale](https://github.com/facebookresearch/fairscale),
with which they trained [Llama](https://github.com/facebookresearch/llama).
MosaicML has recently released [LLM Foundry](https://github.com/mosaicml/llm-foundry), which we used to train
[BioMedLM](https://crfm.stanford.edu/2022/12/15/biomedlm.html) last autumn. Previously, we released [Mistral](
https://github.com/stanford-crfm/mistral) built on [Hugging Face Transformers](https://github.com/huggingface/transformers/)
and [DeepSpeed](https://github.com/microsoft/DeepSpeed).

In the Jax community, there are a number of libraries popping up. Google has released [T5X](https://github.com/google-research/t5x)
and [MaxText](https://github.com/google/maxtext). There are also a number of independent libraries, including [EasyLM](https://github.com/young-geng/EasyLM)
and [JaxSeq](https://github.com/Sea-Snell/JAXSeq), both of which are based on [Flax](https://github.com/google/flax/)
and modified libraries from [Hugging Face Transformers](https://github.com/huggingface/transformers/). Salesforce has
released the Haiku-based [Jaxformer](https://github.com/salesforce/jaxformer). Previously, Eleuther AI released
[mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax/), though it is mostly unmaintained now and uses
older, quasi-deprecated Jax APIs for distributed training.

## A New Codebase for Foundation Model Training

Despite the wide array of existing frameworks, when we started, we found that none of them fully addressed our needs.
At CRFM, we focused on three fundamental goals:

* **Legibility and Composability**: We prioritized writing code that is easy to read, understand, and compose.
* **Reproducibility and Resilience**: We emphasized the ability to reproduce results *exactly*, even in the face of preemption and restarts from checkpoints.
* **Scalability and Efficiency**: We aimed to fully utilize the Cloud TPU resources from Google's TPU Research Cloud program, as well as our own GPU resources, without compromising efficiency.

We chose [Jax](https://github.com/google/jax/) as our framework because it is a powerful, flexible, and performant,
and offers strong reproducibility guarantees. Jax also works well on TPUs, while we found that PyTorch support was still uneven.
Jax is also a natural choice because it allows you to focus on the "what" of your code, and not on the "how": details of
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

Haliax is a library for named tensors, built on Jax and [Equinox](https://github.com/patrick-kidger/equinox),
which is a neural network library for Jax that provides a familiar, PyTorch-like module structure. Haliax uses
Equinox's module structure for its neural network library, rather than Flax or Haiku.

Named Tensors are a powerful abstraction that allow you to give names to the axes of your tensors. These names help
make your code more legible, more composable, and less bug-prone. In Haliax, they also form the basis of how we handle
scale with [Fully-Sharded Data Parallel](https://engineering.fb.com/2021/07/15/open-source/fsdp/) and tensor parallelism.
(See below for our tutorial!)

Haliax is modeled on Alexander Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor),
which argues that named tensors are a better abstraction than the positional axes that are common in deep learning.
In particular, he argues that:

* Named axes are more semantically meaningful and rely less on bitrot-prone comments.
* Named axes allow you to abstract over unreferenced dimensions, making code more flexible.
* Broadcasting leads to unreadable strings of `view`s and `squeeze`s.

To this, I'd add that the implicit broadcasting so common in deep learning code is a source of easy-to-miss bugs, and
that named tensors eliminate many of these bugs.

### A Quick Example: Attention in Haliax

This blog post isn't the place for a full introduction to Haliax (please see the [Haliax tutorial](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC)),
but here's a quick example of a minimal, but full-featured, attention implementation in Haliax.

```python
import jax.numpy as jnp
import haliax as hax

# Named Axes for Tensor Dimensions
Pos = hax.Axis("position", 1024)  # sequence
KPos = Pos.alias("key_position")  # key sequence for attention
Head = hax.Axis("head", 8)  # number of attention heads
Key = hax.Axis("key", 64)  # key/query/value size
Embed = hax.Axis("embed", 512)  # embedding size

def attention(Key, KPos, query, key, value, mask):
    # how similar is each query to each key
    scores = hax.dot(Key, query, key) / jnp.sqrt(Key.size)

    # mask out invalid positions
    if mask is not None:
      scores -= 1E9 * (1.0 - mask)

    # convert to probabilities
    scores = hax.nn.softmax(scores, axis=KPos)

    # weighted sum of values
    return hax.dot(KPos, scores, value)
```

In this example, we've defined an axis for each dimension of our tensors. In Haliax, the named `Axis` is the basic
building block of named tensors, pairing a name with a size. We can then use these axes to define our tensors, and use
those axes to perform operations like `softmax` and tensor multiplication (`dot`).

### Compositionality

Despite making no reference to batching or heads, this same implementation is also batch-capable and supports multi-headed
(or multi-query) attention and even attending to or from non-sequential keys (e.g. attending to image patches):

```python
Batch = hax.Axis("batch", 8)  # batch size

query = hax.random.normal(PRNGKey(0), (Batch, Head, Pos, Key))
key = hax.random.normal(PRNGKey(1), (Batch, Head, KPos, Key))
value = hax.random.normal(PRNGKey(2), (Batch, Head, KPos, Key))

# traditional batched multi-headed attention
assert attention(Key, KPos, query, key, value, mask=None).axes == (Batch, Head, Pos, Key)

# multi-query attention. Each key/value pair produces only one head
key = hax.random.normal(PRNGKey(1), (Batch, KPos, Key))
value = hax.random.normal(PRNGKey(2), (Batch, KPos, Key))
assert attention(Key, KPos, query, key, value, mask=None).axes == (Batch, Head, Pos, Key)

# image patch cross-attention from a sequence
Height = hax.Axis("height", 32)
Width = hax.Axis("width", 32)

key = hax.random.normal(PRNGKey(1), (Batch, Head, Height, Width, Key))
value = hax.random.normal(PRNGKey(2), (Batch, Head, Height, Width, Key))

# KPos in attention actually be a tuple of axes.
assert attention(Key, (Height, Width), query, key, value, mask=None).axes == (Batch, Head, Pos, Key)
```

This compositionality is possible because we've abstracted over the unreferenced dimensions of our tensors.
In the first example, both the `Batch` and `Head` axes are unreferenced, so they are automatically "batched" over.
Similarly, in the second example, we omit the `Head` axis from the `key` and `value` tensors, but attention still works.
In the third example, we can use tuples of axes in many places where we would normally use a single axis.


### Avoiding Bugs

Earlier, I claimed that named tensors can help avoid common bugs. Here's an example of a bug that is easy to make
and hard to spot in a traditional tensor library. Consider the following simple linear model:

```python
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey

# predict y from x using  a linear model (W)
x = jrandom.uniform(PRNGKey(0), (128, 64))
y = jrandom.uniform(PRNGKey(1), (128,))
W = jrandom.uniform(PRNGKey(2), (64, 1))

def mse(pred, target):
    return jnp.mean((pred - target) * (pred - target) )

y_pred = x @ W
mse(y_pred, y)
```

This code appears straightforward, but it has a bug: the dimensions of `y_pred` and `y` are not the same.
Because `y_pred` is a 2D array of shape `(128, 1)`, and `y` is a 1D array of shape `(128,)`, the `-` operator will broadcast `y` to shape `(128, 128)`.
(This makes the subtraction an "outer product"-like operation rather than the intended elementwise subtraction.)
But, you won't get an error at runtime; this is a silent bug. The `mean` call hides the bug by averaging over all values.

This is a common bug in deep learning code, and it's easy to miss. I have personally lost multiple days to this exact bug over the years,
in every deep learning framework I've used.

But if we use named tensors, we avoid this bug without having to think about it. Here's what this code looks like in Haliax:

```python
import haliax as hax
from jax.random import PRNGKey

Batch = hax.Axis("batch", 128)
Feature = hax.Axis("feature", 64)

x = hax.random.uniform(PRNGKey(0), (Batch, Feature))
y = hax.random.uniform(PRNGKey(1), Batch, 0, 2)

def mse(pred, target):
    return hax.mean((pred - target) * (pred - target), axis=Batch)

W = hax.random.uniform(PRNGKey(2), (Feature,))

y_pred = hax.dot(Feature, x, W)
mse(y_pred, y)
```

The code is basically the same, but the presence of named axes mean that we don't accidentally broadcast `y` to the wrong shape.
Instead, it works exactly as we intend.

### Scale via Named Tensors

We use named axes both to improve legibility and to enable scale: named axes are the basis of our
[Fully-Sharded Data Parallel](https://engineering.fb.com/2021/07/15/open-source/fsdp/) implementation as well as for tensor parallelism.
FSDP can be added to a training loop with about 10 lines of code, enabling scale to at least 256 TPU cores (which is
as many as we can get our hands on) and at least 65B parameters (which is way bigger than we have compute for).

FSDP with Haliax basically amounts to telling Haliax which axes to shard, and specifying a different sharding for computation than for storage.
A full tutorial is available [here](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz?usp=sharing), but here's a quick example:

```diff
+# describe how we shard our parameters and our data
+# We store our parameters and optimizer states fully sharded along the embed axis
+param_mapping = {"embed": "data"}
+# During computation, we instead shard our data along the batch axis, and gather the parameters just-in-time
+data_mapping = {"batch": "data"}

+# tell Haliax to shard our model and optimizer states
+@hax.named_jit
def init_model():
-    return MyModel()
+    return hax.shard_with_axis_mapping(MyModel(), param_mapping)

model = init_model()

# initialize optimizer
import optax
optimizer = optax.adamw(1E-4, weight_decay=0.1)

+@hax.named_jit
def init_optimizer(model):
    opt_state = optimizer.init(model)
-    return opt_state
+    return hax.shard_with_axis_mapping(opt_state, param_mapping)

optimizer = init_optimizer(model)

+@hax.named_jit
def train_step(model, opt_state, input_ids):
  ... # elided for brevity

  # ensure that intermediate states are sharded correctly
-  loss, grads = grad_loss(model, input_ids)
+  with hax.axis_mapping(data_mapping):
+    loss, grads = grad_loss(model, input_ids)

  ...
  return loss, model, opt_state

for data in data_iter:
+  data = hax.shard_with_axis_mapping(data, data_axis_mapping)
  ...
```

Tensor parallelism can be added by simply changing the two axis mappings:

```diff
# Specify which axes we shard for tensor parallelism:
# specifying "head" shards attention and "mlp" shards the feedforward
+tensor_parallel_mapping = {"head": "model", "mlp": "model"}
# We store our parameters and optimizer states fully sharded along the embed axis
-param_mapping = {"embed": "data"}
+param_mapping = {"embed": "data", **tensor_parallel_mapping}
# During computation, we instead shard our data along the batch axis, and gather the parameters just-in-time
-data_mapping = {"batch": "data"}
+data_mapping = {"batch": "data", **tensor_parallel_mapping}
```

This is all that is required to shard a model across multiple GPUs or TPUs. The rest of the training loop remains unchanged.
You can do fancier things like sharded data loading (which we do in Levanter), but the basic idea is the same.


### Haliax Tutorials

This is just a taste of what Haliax can do. For more details, please see our interactive tutorials on Colab:

* [Introduction to Haliax with Transformers](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC?usp=sharing)
* [Scaling Transformers in Haliax](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz?usp=sharing), including FSDP in Jax.

## Levanter: Bitwise Reproducible Foundation Models with Jax

Levanter is a library for training foundation models built on top of Haliax. It provides a complete pipeline
for training a GPT-2-like Transformer, complete with data preparation, logging, training, checkpointing, evaluation, and
export, while maintaining bitwise reproducibility throughout.

We have used Levanter to train models as large as 6.7b parameters on a v3-256, and have run experiments showing that it
can scale up to least 65b parameters.

### Bitwise Reproducibility

One of the benefits of Jax is that it offers strong guarantees for reproducibility. In particular, Jax's fine-grained
control over random number generation makes it easy to ensure bitwise reproducibility, especially when using TPUs.
Levanter takes advantage of this to offer bitwise reproducibility for training runs, even after preemption. In particular,
the same run on the same set of hardware (e.g. a v3-32 or a v3-256) will produce the exact same loss curve, even if it is
preempted and resumed multiple times. As an example, here is a screenshot of a training run being resumed multiple times, even on different TPU pod slices:

![plot showing bitwise reproducibility with four training runs superimposed with the exact same loss curve](figures/bitwise_repro_curve.png)

The fact that you can't make out the different lines is the point: the training runs are bitwise identical,
a huge advantage for debugging and reproducibility.

### Efficiency and Scale

XXX something something v3-256 scaling numbers?

### Other Features

* **Preprocessing**: Levanter uses [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers/) to preprocess text
using distributed preprocessing backed by [Ray](https://www.ray.io/).
* **Training**: In addition to Jax and Haliax, Levanter uses [Optax](https://github.com/deepmind/optax) for optimization.
  (though our new optimizer, [Sofia](https://arxiv.org/abs/2305.14342), is coming to Levanter soon!)
* **Logging**: Logging is done with [WandB](https://wandb.ai/), complete with a fancy online visualization of the validation set during training.
* **Export**: We also support exporting models to the Hugging Face Hub, with export compatible with Pytorch and Transformers via [SafeTensors](https://github.com/huggingface/safetensors).
* **Checkpointing**: Distributed checkpointing is supported via Google's [TensorStore](https://google.github.io/tensorstore/) library and transparently supports exporting checkpoints from a single machine.
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


#### On-Demand Data Preprocessing

Levanter can automatically spin up a Ray cluster using the nodes being used for training, using the typically impressive CPUs of those machines to preprocess data. This is especially useful for large data sets
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
