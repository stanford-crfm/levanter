# Levanter

> You could not prevent a thunderstorm, but you could use the electricity; you could not direct the wind, but you could trim your sail so as to propel your vessel as you pleased, no matter which way the wind blew. <br/>
> — Cora L. V. Hatch


Levanter is a library based on [Jax](https:://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox)
for training [foundation models](https://en.wikipedia.org/wiki/Foundation_models) created by [Stanford's Center for Research
on Foundation Models (CRFM)](https://crfm.stanford.edu/).

## Haliax

> Though you don’t seem to be much for listening, it’s best to be careful. If you managed to catch hold of even just a piece of my name, you’d have all manner of power over me.<br/>
> — Patrick Rothfuss, *The Name of the Wind*

Haliax is a module (currently) inside Levanter for named tensors, modeled on Alexander Rush's [Tensor Considered Harmful](https://arxiv.org/abs/1803.09868).
It's designed to work with Equinox to make constructing distributed models easier.
