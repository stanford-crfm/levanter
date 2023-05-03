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
    <strong> We introduce our new project
        <a href="https://github.com/stanford-mercury/levanter" target="_blank">Levanter</a>, our code
        and infrastructure for training foundation models models using Jax. We also XXX checkpoints.
    </strong>
> </div>


## Introduction


 our new Jax-based codebase for training foundation models. Levanter depends on
a lot of great libraries from the community, including [Equinox](XXX), [Optax](XXX), [Hugging Face Transformers](XXX)


## Why another project?

In a [previous post](https://crfm.stanford.edu/2021/08/26/mistral.html), we announced [Mistral](https://github.com/stanford-mercury/mistral),
a project for training moderately-sized GPT-2 models using PyTorch of a few hundred million parameters. Mistral is great
and we still recommend Mistral for many use cases. XXX

As a university, we depend on the generosity of cloud providers and other funders to support our research. Compute
can come from a variety of sources: our own internal clusters, cloud providers, and research grants. Google in particular
has been a great supporter of our research, providing us with access to TPUs and TPU pods through the
[TPU research cloud](https://sites.research.google/trc/about/). We were excited to try out the TPU pods, but we quickly
ran into a problem: PyTorch support for TPU pods is still in its infancy. Despite a lot of hard work from the Pytorch-XLA
team, our initial experience was a string of bugs and limitations. It is a heroic undertaking to match the highly dynamic
nature of PyTorch with the highly static nature of TPUs/XLA. In particular, multi-host training, necessary for training large models, was not yet well-supported.

We were also excited to try out [Jax](https://github.com/google/jax), which has been gaining popularity in the ML community.
Jax is a Python library built on top of XLA that provides a NumPy-like interface for writing high-performance code that
can be run on TPUs, GPUs, and CPUs. Jax is a great fit for our diverse compute environment. Jax also offers
strong guarantees for reproducibility; using Jax, we are able to offer bitwise determinism for our models (given
the same compute), which is important for debugging and reproducibility.

That leaves the question of why not [T5X](https://github.com/google-research/t5x), which is Google's own Jax-based
codebase for training LLMs. We found that T5X was not a great fit for our use cases. T5X is pretty deeply integrated
into the Google ecosystem, which makes it less suitable to the Hugging Face-oriented ecosystem that we and many others
use. We also wanted to develop a code base that felt a bit lighter weight and more flexible than T5X, with an emphasis
on allowing graduate students to iterate on new ideas quickly. Finally, we wanted to learn more about training large
foundation models, and we felt that building our own codebase would be a good way to do that.


## Levanter Rises

So we began work on new codebase, which we call Levanter. Levanter is a Jax-based codebase for training foundation models
that is designed to be flexible, modular, and accessible, while still being performant and scalable. In particular,
we want Levanter to be an easy and natural way for graduate students and other researchers to experiment with new ideas
at larger scales than they might otherwise be able to. We also want Levanter to be a useful reference for others who
are interested in building their own codebases for training foundation models.

Levanter offers:
* A modular, extensible codebase for training foundation models
* Easy, customizable support for Fully-Sharded Data Parallelism (FSDP) and tensor parallelism
* Support for resuming from preemption (important when using donated compute)
* Bitwise reproducibility of training runs, even with preemption
* Distributed, just-in-time preprocessing
* Built-in integrations with Wandb and Hugging Face Datasets and Hugging Face Tokenizers
* Export of models to PyTorch State Dicts, Hugging Face's new safetensors library (XXX), and the HF model hub

Levanter is built on top of a number of great libraries from the community, including [Equinox](XXX), [Optax](XXX),
[Hugging Face Transformers](XXX), and [Wandb](XXX). Levanter is still a work in progress, but we are excited to share it
with the community. We hope that Levanter will be useful to others who are interested in training foundation models
using Jax.

### Features

#### Improvements over Mistral

Levanter is a successor to Mistral, and we have made a number of improvements. In particular, Levanter offers:

* TPU support (naturally)
* Flexible tensor parallelism support
* Bitwise reproducibility
* Improved support for resuming from preemption
* Fancy visualization of token probabilities integrated into Wandb during training to understand how models are learning and identify issues in the data
* (Coming Soon) Simultaneous preprocessing and training

Levanter is not yet at complete feature parity. Beyond the obvious lack of PyTorch support, Levanter does not yet support
multi-machine training with GPUs, due to some issues with CUDA configuration we haven't quite worked out yet.

#### FSDP and Tensor Parallelism with Named Axes

Levanter is built on top of Haliax, a new and, for now, bundled library that uses named axes, in the style of Alexander
Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor) (which has since been adopted into PyTorch).
Named axes offer a number of advantages over the more traditional approach of using positional axes. To rehash some of
the arguments:

* Implicit broadcasting is a source of bugs
* Positional axes offers less flexibility in terms of adding new axes (Jax's `vmap` is a notable exception, but has its limits)
* Named axes are more semantically meaningful and rely less on bitrot-prone comments

I in particular have lost more time than I care to admit to bugs caused by implicit broadcasting. Named axes are a great
way to avoid these bugs.

Moreover, named axes have yet another advantage: they make it even easier to define custom tensor-parallel and FSDP schemes
in Jax. Using named axes with Haliax, FSDP is as simple as asking that the model's parameters along with optimizer states
be sharded along a particular axis (typically the "embed" or "hidden" axis in transformers.) Tensor parallelism is
just as simple. For more details, please see the [Architectural Overview](XXX) in the Levanter documentation.

#### Bitwise Reproducibility

One of the benefits of Jax is that it offers strong guarantees for reproducibility. In particular, Jax's fine-grained
control over random number generation makes it possible to offer bitwise reproducibility, especially when using TPUs.
Levanter takes advantage of this to offer bitwise reproducibility for training runs, even after preemption. As an example,
here is a screenshot of a training run being resumed multiple times, even on different TPU pod slices:

![plot showing bitwise reproducibility with four training runs superimposed with the exact same loss curve](figures/bitwise_repro_curve.png)

The fact that you can't make out the different lines is the point here: the training runs are bitwise identical,
a huge advantage for debugging and reproducibility.


#### Token Probability Visualization

As we've been working with teams who are building models with their own domain-specific models, data preparation has
been a significant challenge. In particular, it can be difficult to identify issues in the data that are causing
particularly low loss. As a small step towards addressing this, we have added a feature to Levanter that allows you
to visualize the probability of each token in a sample of the validation set during training. For novel data sets,
this allows us to identify, e.g., highly redundant data (what we call "madlib duplicates") or other issues that might
be causing low loss. Here is an example of the token probability visualization in action:

XXX legal data?

This visualization is logged to WandB as training progresses, which is a nice alternative to just staring at the loss
curve, which is what I usually do.

## Released Models

Along with the release of the code, we are releasing a few models trained using Levanter. These models are available on
the [HF model hub](XXX) and can be used with the Hugging Face Transformers library. We have more in development and will
be releasing them as they become available.

XXX TODO: verify this is what we're releasing, add links
- GPT-2-1536 (1.45B parameters) XXX, including checkpoints every 10,000 steps
- XXX music model (link to other blog post XXX)
- Backpacks? XXX

## Future and Conclusion

This is just the beginning for Levanter.

In the future, look for:
* more models on interesting problem domains
* scaled up versions of new architectures developed here at Stanford and elsewhere
* new training techniques
* larger models

We are excited to continue to develop Levanter and to share our work with the
community. Please join us on our journey! (And by the way, [we're hiring](https://crfm.stanford.edu/apply.html)!)

## Acknowledgements

In addition to the generous support of Google, we would like to thank the following people for their help and support:

* Sidd Karamcheti for support and XXX
* Yifan Mai, Tony Lee, Jason Bolton, Ivan Zhou, and the rest of the CRFM engineering team for support and discussions.
* The TRC team for support and help
* Roy Froystig, Sholto Douglas, and the rest Jax team for help with debugging and just for making JAx.
