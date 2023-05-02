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
that is designed to be flexible, modular, and accessible, while still being performant and scalable. Levanter is built
on top of a number of great libraries from the community, including [Equinox](XXX), [Optax](XXX), [Hugging Face Transformers](XXX), and [Wandb](XXX).

Levanter offers:
* A modular, extensible codebase for training foundation models
* Easy, customizable support for Fully-Sharded Data Parallelism (FSDP) and tensor parallelism
* Bitwise reproducibility of training runs, even with preemption
* Support for resuming from preemption (important when using donated compute)
* Distributed, just-in-time preprocessing
* Built-in integrations with Wandb and Hugging Face Datasets and Hugging Face Tokenizers
* Export of models to PyTorch State Dicts, Hugging Face's new safetensors library (XXX), and the HF model hub

Levanter is still a work in progress, but we are excited to share it with the community. We hope that Levanter will
be useful to others who are interested in training foundation models using Jax. We also hope that Levanter will be


### Improvements over Mistral

Levanter is a successor to Mistral, and we have made a number of improvements over Mistral. In particular, Levanter
offers:

* TPU support (naturally)
* Bitwise reproducibility
* Fancy visualization of token probabilities integrated into Wandb during training to understand how models are learning and identify issues in the data
* Flexible tensor parallelism support
* Improved support for resuming from preemption

### Under the hood: Haliax

Levanter is built on top of Haliax, a new and, for now, bundled library that uses named axes, in the style of Alexander
Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor) (which has since been adopted into PyTorch).
Named axes offer a number of advantages over the more traditional approach of using positional axes. To rehash some of
Sasha's arguments:

* Implicit broadcasting is a source of bugs
* Positional axes offers less flexibility in terms of adding new axes (Jax's `vmap` is a notable exception, but has its limits)
* Named axes are more semantically meaningful and rely less on bitrot-prone comments

I in particular have lost more time than I care to admit to bugs caused by implicit broadcasting. Named axes are a great
avoid these bugs.

Moreover, named axes have yet another advantage: they make it much easier to define and custom tensor-parallel and FSDP schemes
in Jax. Using named axes with Haliax, FSDP is as simple as asking that the model's parameters along with optimizer states
be sharded along a particular axis (typically the "embed" or "hidden" axis in transformers.) Tensor parallelism is
just as simple. For more details, please see the [Architectural Overview](XXX) in the Levanter documentation.



## Released Models

Along with the release of the code, we are releasing a few models trained using Levanter. These models are available on
the [HF model hub](XXX) and can be used with the Hugging Face Transformers library. We have more in development and will
be releasing them as they become available.

XXX TODO: verify this is the one we're releasing
- GPT-2-1536 (1.45B parameters) XXX, including checkpoints every 10,000 gradient steps
- XXX music model
- Backpacks?

## Future and Conclusion

This is just the beginning for Levanter

In the future, look for:
* more models on interesting problem domains
* scaled up versions of new architectures developed here at Stanford and elsewhere
* training techniques
* larger models

## Acknowledgements

In addition to the generous support of Google, we would like to thank the following people for their help and support:

* Sidd Karamcheti for support and XXX
* Yifan Mai, Tony Lee, Jason Bolton, Ivan Zhou, and the rest of the CRFM engineering team for support and discussions.
* Roy Froystig, Sholto Douglas, and the rest Jax team for help with debugging and just for making JAx.
