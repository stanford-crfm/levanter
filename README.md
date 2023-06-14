# Levanter and Haliax

Levanter and Haliax are libraries based on [Jax](https:://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox)
for training deep learning models, especially [foundation models](https://en.wikipedia.org/wiki/Foundation_models).
Haliax is a named tensor library (modeled on [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor)) that focuses on improving the legibility and compositionality of deep learning code while still maintaining efficiency and scalability.
Levanter is a library for training foundation models built on top of Haliax.
In addition to the goals of legibility, efficiency, and scalability, Levanter further strives for bitwise reproducibility,
meaning that the same code with the same data will produce the exact same result, even in the presence of preemption and restarting from checkpoints.

Levanter and Haliax were created by [Stanford's Center for Research on Foundation Models (CRFM)](https://crfm.stanford.edu/)'s research engineering team. ([We're hiring!](https://crfm.stanford.edu/apply.html))

## Haliax

> Though you don’t seem to be much for listening, it’s best to be careful. If you managed to catch hold of even just a piece of my name, you’d have all manner of power over me.<br/>
> — Patrick Rothfuss, *The Name of the Wind*

Haliax is a Jax library for building neural networks with named tensors, in the tradition of Alexander Rush's [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor).
We use named tensors to improve the legibility and compositionality of our programs without sacrificing performance or scalability.

Here's a minimal attention module implementation in Haliax. For a more detailed introduction, please see the [Haliax tutorial](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC).

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
import haliax.nn as hnn

Pos = hax.Axis("position", 1024)  # sequence length
KPos = Pos.alias("key_position")
Head = hax.Axis("head", 8)  # number of attention heads
Key = hax.Axis("key", 64)  # key size
Embed = hax.Axis("embed", 512)  # embedding size

def attention_scores(Key, KPos, query, key, mask):
  # how similar is each query to each key
  scores = hax.dot(Key, query, key) / jnp.sqrt(Key.size)

  if mask is not None:
    scores -= 1E9 * (1.0 - mask)

  # convert to probabilities
  scores = hax.nn.softmax(scores, KPos)
  return scores


def attention(Key, KPos, query, key, value, mask):
  scores = attention_scores(Key, KPos, query, key, mask)
  answers = hax.dot(KPos, scores, value)

  return answers

# Causal Mask means that if pos >= key_pos, then pos can attend to key_pos
causal_mask = hax.arange(Pos).broadcast_axis(KPos) >= hax.arange(KPos)

class Attention(eqx.Module):
  proj_qkv: hnn.Linear  # input projection from [Embed] -> [(q, k, v), Head, Key]
  proj_answer: hnn.Linear  # output projection from [Head, Key] -> [Embed]

  @staticmethod
  def init(Embed, Head, Key, *, key):
    Qkv = hax.Axis("qkv", 3)  # create all three at once

    k_qkv, k_ans = jax.random.split(key, 2)
    proj_qkv = hnn.Linear.init(In=Embed, Out=(Qkv, Head, Key), key=k_qkv)
    proj_answer = hnn.Linear.init(In=(Head, Key), Out=Embed, key=k_ans)
    return Attention(proj_qkv, proj_answer)

  def __call__(self, x, mask=None):
    qkv_out = self.proj_qkv(x)
    q, k, v = qkv_out.unbind("qkv")

    # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
    k = k.rename({"position": "key_position"})
    v = v.rename({"position": "key_position"})

    answers = attention(Key, KPos, q, k, v, causal_mask)

    x = self.proj_answer(answers)
    return x
```

### Documentation for Haliax

Currently we have two tutorials for Haliax:
* [Introduction to Haliax with Transformers](https://colab.research.google.com/drive/1TiTcQQ4V5mopbgCu1SVl-oqJtXn7rFnC)
* [Distributed Training in Haliax](https://colab.research.google.com/drive/1QX4yH3zRFF3Xiibf1aahETcSQ5nbcUMz) (including FSDP)

## Levanter

> You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew. <br/>
> — Cora L. V. Hatch

Levanter is a library for training foundation models built on top of Haliax. Levanter strives for bitwise reproducibility,
meaning that the same code with the same data will produce the exact same result, even in the presence of preemption and restarting from checkpoints.
It supports distributed training on TPUs (and, soon, GPUs), including FSDP, tensor parallelism, distributed checkpointing, distributed data loading, and more.
Levanter integrates with WandB for logging and with the Hugging Face ecosystem for tokenizers, datasets, and model import and export.

### Installing Levanter

First, install the appropriate version of Jax for your system. See [Jax's installation instructions](https://github.com/google/jax/blob/main/README.md#installation) as it varies from platform to platform.

If you're using a TPU, more complete documentation for setting that up is available [here](docs/Getting-Started-TPU-VM.md). GPU support is still in-progress; documentation is available [here](docs/Getting-Started-CUDA.md).

Now clone this repository and install it with pip:

```bash
git clone https://github.com/stanford-crfm/levanter.git
cd levanter
pip install -e .
wandb login  # optional, we use wandb for logging
```

TODO: put things on pypi, etc

### Training a GPT2-nano

As a kind of hello world, here's how you can train a GPT-2 "nano"-sized model on a small dataset.

```bash
python src/levanter/main/train_lm.py --config_path config/gpt2_nano.yaml
```

This will train a GPT2-nano model on the [WikiText-2](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset.
You can change the dataset by changing the `dataset` field in the config file.

For more information about the various configuration options, please see the [Training Getting Started](docs/Getting-Started-Training.md) guide.
You can also use `--help` or poke around other configs to see all the options available to you.



#### Training on a TPU Cloud VM

Please see the [TPU Getting Started](docs/Getting-Started-TPU-VM.md) guide for more information on how to set up a TPU Cloud VM and run Levanter there.

#### Training with CUDA

Please see the [CUDA Getting Started](docs/Getting-Started-CUDA.md) guide for more information on how to set up a CUDA environment and run Levanter there.

### Understanding Levanter and Haliax

Please see the [Overview](docs/Overview.md) guide for more information on how Levanter and Haliax work, their inspirations, etc.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

Levanter is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
