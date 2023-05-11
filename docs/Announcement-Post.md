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
    <strong> We introduce our project
        <a href="https://github.com/stanford-mercury/levanter" target="_blank">Levanter</a>, our code
        and infrastructure for training foundation models models using Jax. We also XXX checkpoints.
    </strong>
> </div>


## Introduction

XXX really struggling with intro Here's a crappy cut:

Foundation models have revolutionized artificial intelligence. By scaling up the size of models, researchers have
been able to train models that are capable of performing a wide range of tasks, including tasks that a few years ago
seemd out of reach of AI systems. However, training these models is a challenge. It requires a lot of compute - the
largest models can require hundreds of GPU-years of compute. It is also difficult to reproduce results, since even
small changes in the training process can lead to large changes in performance.

At the [Center for Research on Foundation Models](https://crfm.stanford.edu), we have developed a new code base for
training foundation models, which we call [Levanter](https://github.com/stanford-crfm/levanter). Levanter is a
[Jax](https://github.com/google/jax)-based codebase for training foundation models that is designed to be flexible,
modular, and accessible, while still being performant and scalable. In particular, Levanter is designed to be able to
train models on a variety of different hardware, including GPUs, TPUs, and TPU pods. Levanter also offers strong
guarantees about reproducibility, and we have released checkpoints for a number of models trained with Levanter.

Together with [HELM](https://crfm.stanford.edu/helm/latest/), our evaluation framework, Levanter makes it easy to
produce reproducible results for foundation models. We hope that Levanter will be a useful tool for the community.

## Why another project?

In a [previous post](https://crfm.stanford.edu/2021/08/26/mistral.html), we announced [Mistral](https://github.com/stanford-mercury/mistral),
a project for training moderately-sized GPT-2 models using PyTorch of a few hundred million parameters. Mistral is a
great library for training models of that scale. We still recommend Mistral for those use cases. So, why did we create
Levanter?

As a university, we depend on the generosity of supporters to fund our research. Compute can come from a variety of
sources: our own internal clusters, cloud providers, and research grants. In particular this means that there is no
single compute provider that we can depend on. We need to be able to train our models on a variety of different hardware.
Google in particular has been a great supporter of our research, providing us with access to TPU pod slices through the
[TPU research cloud](https://sites.research.google/trc/about/). We were excited to try out the TPU pods, but we quickly
ran into a problem: PyTorch support for TPU pods is still in its infancy. It is a heroic undertaking to match the highly
dynamic nature of PyTorch with the highly static nature of TPUs/XLA, but, despite a lot of hard work from the Pytorch-XLA
team, our initial experience was a string of bugs and limitations.  In particular, multi-host training, necessary for
training large models, was not yet well-supported.

We were also excited to try out [Jax](https://github.com/google/jax), which has been gaining popularity in the ML community.
Jax is a Python library built on top of XLA that provides a NumPy-like interface for writing high-performance code that
can be run on TPUs, GPUs, and CPUs. Jax is a great fit for our diverse compute environment. Jax also offers strong
guarantees for reproducibility; using Jax, we are able to offer bitwise determinism for our models (given
the same compute), which is important for debugging and reproducibility.

That leaves the question of why not [T5X](https://github.com/google-research/t5x), which is Google's own Jax-based
codebase for training LLMs. We found that T5X was not a great fit for our use cases. T5X is pretty deeply integrated
into the Google ecosystem, which makes it less suitable to the Hugging Face-oriented ecosystem that we and many others
use. We also wanted to develop a code base that felt a bit lighter weight and more flexible than T5X, with an emphasis
on allowing graduate students to iterate on new ideas quickly. Finally, we wanted to learn more about training large
foundation models, and we felt that building our own codebase would be a good way to do that.

XXX There are a few other libraries XXX


## Levanter Rises

So we began work on new codebase, which we call Levanter. Levanter is a Jax-based codebase for training foundation models
that is designed to be flexible, modular, and accessible, while still being performant and scalable. In particular,
we want Levanter to be an easy and natural way for graduate students and other researchers to experiment with new ideas
at larger scales than they might otherwise be able to. We also want Levanter to be a useful reference for others who
are interested in building their own codebases for training foundation models.

Levanter offers:
* A modular, extensible codebase for training foundation models
* Easy, customizable support for [Fully-Sharded Data Parallelism (FSDP)](https://engineering.fb.com/2021/07/15/open-source/fsdp/) and tensor parallelism
* Support for resuming from preemption (important when using donated compute)
* Bitwise reproducibility of training runs, even with preemption
* (Coming soon) Distributed, just-in-time preprocessing
* Built-in integrations with Wandb and Hugging Face Datasets and Hugging Face Tokenizers
* Export of models to PyTorch state dict serialization format, Hugging Face's new safetensors library (XXX), and the HF model hub
* Visualization of token probabilities integrated into WandB during training

Levanter is built on top of a number of great libraries from the community, including [Equinox](https://github.com/patrick-kidger/equinox), [Optax](https://github.com/deepmind/optax),
[Hugging Face Transformers](https://github.com/huggingface/transformers), and [WandB](https://wandb.ai). Equinox is a simple but powerful library for organizing neural
network modules in Jax, and Optax is a library for defining and applying optimizers in Jax. I believe Hugging Face
Transformers and WandB need no introduction.

Levanter is still a work in progress, but we are excited to share it with the community. We hope that Levanter will be useful to others who are interested in training foundation models
using Jax and TPUs.

### Features

#### Improvements over Mistral

Levanter is a successor to Mistral, and we have made a number of improvements. In particular, Levanter offers:

* TPU support (naturally)
* Flexible tensor parallelism support
* Bitwise reproducibility
* Improved support for resuming from preemption
* (Coming Soon) Simultaneous, distributed preprocessing and training

Levanter is not yet at complete feature parity with Mistral. Beyond the obvious lack of PyTorch support, Levanter does
not yet support multi-machine training with GPUs, due to [some issues with CUDA configuration](https://github.com/stanford-crfm/levanter/issues/113) we haven't quite worked out
yet. Some libraries (e.g. WandB) have better PyTorch integration than Jax integration, so some features aren't quite
as polished as they are in Mistral. We are working on improving these issues.

#### FSDP and Tensor Parallelism with Named Axes

Levanter is built on top of Haliax, a new and, for now, bundled library that uses named axes, in the style of Alexander
Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor). Named axes offer a number of advantages
over the more traditional approach of using positional axes. To rehash some of the arguments:

* Named axes are more semantically meaningful and rely less on bitrot-prone comments.
* Implicit broadcasting is a source of bugs.
* Positional axes offers less flexibility in terms of adding new axes (Jax's `vmap` is a notable exception, but has its limits).

I in particular have lost more time than I care to admit to bugs caused by implicit broadcasting. Named axes are a great
way to avoid these bugs.

Here's what attention looks like with named axes:

```python
import haliax as hax
import jax.numpy as jnp
from typing import Optional
from haliax import NamedArray, Axis, AxisSelection

def dot_product_attention(
        HeadDim: Axis,
        KeyPosition: AxisSelection,
        query: NamedArray,
        key: NamedArray,
        value: NamedArray,
        mask: Optional[NamedArray] = None) -> NamedArray:
    query = query / jnp.sqrt(HeadDim.size)

    weights = hax.dot(HeadDim, query, key)

    if mask is not None:
        weights = hax.where(mask, weights, -1e9)

    weights = hax.nn.softmax(weights, axis=KeyPosition)
    return hax.dot(KeyPosition, weights, value)
```

And you can call it like so:
```python
    q: NamedArray # [Batch, Pos, HeadDim]
    k, v: NamedArray # [Batch, KPos, HeadDim]
    attn = dot_product_attention(HeadDim, KPos, q, k, v)  # [Batch, Pos, HeadDim]
```

I find this code much more readable than the equivalent code using positional axes. Moreover, named axes are more
flexible. Do you want multi-head dot product attention? You don't have to do anything, just call the function.
Or maybe you want to, say, support dot product attention over image patches? you... still don't have to do anything, just
call the function with two "Position" axes:

```python
    q: NamedArray # [Batch, Head, Height, Width, HeadDim]
    k, v: NamedArray # [Batch, Head, KHeight, KWidth, HeadDim]
    attn = dot_product_attention(HeadDim, (KHeight, KWidth), q, k, v)  # [Batch, Head, Height, Width, HeadDim]
```

Moreover, named axes have yet another advantage: they make it even easier to scale up to large models. In particular,
we can use named axes (and Jax's `pjit` sharding) to implement tensor parallelism and FSDP in a way that is easy to use.
With Levanter and Haliax, FSDP is as simple as specifying an axis (typically the "embed" or "hidden" axis for transformers)
that you want to keep your parameters and optimizer states sharded along. We then distinguish between two "configurations"
of your parameters: one for storage (`parameter_axis_mapping`) and one for compute (`compute_axis_mapping`).

For example, here's what your training script might look like:


```python
  def compute_loss(model, data):
    ...


  @named_pjit(axis_resources=trainer.parameter_axis_mapping)
  def train_step(opt_state, model, data):
      with hax.axis_mapping(trainer.compute_axis_mapping):
          loss, grads = jax.value_and_grad(compute_loss)(model, data)
      updates, opt_state = optimizer.update(grads, opt_state, params=model)
      model = eqx.apply_updates(model, updates)
      return loss, model, opt_state
```

Under the hood, all we do is use Jax's `pjit` operator to shard the parameters and optimizer states along the axis
you specify. We then use `hax.axis_mapping` to change the preferred mapping for computing the loss and gradients. Jax
handles most of the heavy lifting, except Linear layers need a bit of handholding, which Haliax provides.

Tensor Parallelism works in a broadly similar manner, with you specifying how "wide" you want model parallelism to be
and specifying which axes of your model should be sharded. (For a transformer, this would typically be the Head
and MLP axes.)

Through our experiments, we have found that FSDP provides sufficient scaling for the model sizes we can feasibly train
using our available resources, particularly on TPU. In practical applications, using FSDP, we observed that tensor
parallelism offered limited utility in training transformers with up to 20 billion parameters. (Tensor parallelism is
useful for inference, but Levanter's focus is on training.)

For (much) more detail, please see the [Architectural Overview](XXX) in the Levanter documentation.

#### Bitwise Reproducibility

One of the benefits of Jax is that it offers strong guarantees for reproducibility. In particular, Jax's fine-grained
control over random number generation makes it easy to ensure bitwise reproducibility, especially when using TPUs.
Levanter takes advantage of this to offer bitwise reproducibility for training runs, even after preemption. As an example,
here is a screenshot of a training run being resumed multiple times, even on different TPU pod slices:

![plot showing bitwise reproducibility with four training runs superimposed with the exact same loss curve](figures/bitwise_repro_curve.png)

The fact that you can't make out the different lines is the point: the training runs are bitwise identical,
a huge advantage for debugging and reproducibility.

#### Token Probability Visualization

As we've been working with teams who are building models with their own domain-specific models, data preparation has
been a significant challenge. In particular, it can be difficult to identify issues in the data that are causing
particularly low loss. As a small step towards addressing this, we have added a feature to Levanter that allows you
to visualize the probability of each token in a sample of the validation set during training. For novel data sets,
this has allowed us to identify, e.g., highly but not perfectly redundant data (what we call "madlib duplicates") or other
issues that might be causing abnormally low loss. We've also used it to qualitative assess how alternative architectures
learn differently from Transformers.

Here is an example of the token probability visualization in action:

![plot showing heat map of token probabilities for a sample of the validation set](figures/token_probabilities.png)

This visualization is logged to WandB as training progresses, which is a nice alternative to just staring obsessively at
the loss curve, which is what I usually do.

## Released Models

Along with the release of the code, we are releasing a few models trained using Levanter. These models are available on
the [HF model hub](XXX) and can be used with the Hugging Face Transformers library. We have more in development and will
be releasing them as they become available.

XXX TODO: verify this is what we're releasing, add links
- GPT-2-1536 (1.45B parameters) XXX, including checkpoints every 10,000 steps
- XXX music model (link to other blog post XXX)
- Backpacks? XXX

## Future and Conclusion

This is just the beginning for Levanter. In the future, look for:
* more models on interesting problem domains
* scaled up versions of new architectures developed here at Stanford and elsewhere
* new training techniques
* larger models

We are excited to continue to develop Levanter and to share our work with the community. Please join us on our journey!
You can find us on [GitHub](https://github.com/stanford-crfm/levanter), [Twitter](https://twitter.com/StanfordCRFM),
and [Discord](https://discord.gg/8JXaqtq7HH). (And by the way, [we're hiring](https://crfm.stanford.edu/apply.html)!)

## Acknowledgements

In addition to the generous support of Google, we would like to thank the following people for their help and support:

* John Thickstun, Sidi Lu, John Hewitt, and others for being early adopters and providing feedback. We really appreciate your patience, support, and feedback.
* Yifan Mai, Tony Lee, Jason Bolton, Ivan Zhou, and the rest of the CRFM engineering team for support and discussions.
* The TRC team for support getting spun up on TPUs and for making the TPU slices available to us.
* Roy Froystig, Sholto Douglas, and the rest Jax team for help with debugging and support.
* Sidd Karamcheti for support and conversations
