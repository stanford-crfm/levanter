# GPT-OSS in Levanter

This document captures the design and implementation notes for porting Hugging Face's `hf_gpt_oss.py` (a PyTorch model) to the Levanter/JAX ecosystem.

## Overview

GPT-OSS is a decoder-only transformer that combines rotary-position attention with a sparse Mixture-of-Experts (MoE) MLP. The implementation here mirrors the architecture from the reference PyTorch model while adopting Levanter conventions:

* **Named axes** via [Haliax](https://github.com/stanford-crfm/haliax) for shape safety.
* **Equinox modules** to define parameterized layers.
* **Functional JAX style** with explicit PRNG keys.

## Components

### Configuration (`GptOssConfig`)

Extends `MistralConfig` and adds MoE and routing options:

* `num_local_experts` – number of experts per MoE layer.
* `num_experts_per_tok` – how many experts a token is routed to.
* `sliding_window` – if set, some layers use sliding‑window attention.
* `router_aux_loss_coef` – weight for the load‑balancing loss.
* Axis accessors: `Experts`, `TopExperts`, etc.

### Router (`GptOssRouter`)

A linear projection over the embedding dimension that scores each expert. The top‑k experts and their normalized routing weights are selected per token using `jax.lax.top_k`.

### Experts (`GptOssExperts`)

Implements the gated GLU variant from the PyTorch model:

1. `gate_up_proj` → split into `gate` and `up` parts.
2. Clamp activations to `[-limit, limit]` (limit = 7.0).
3. Apply gated GELU (`gate * sigmoid(alpha * gate)`, alpha = 1.702).
4. Multiply by `(up + 1)` and project down with `down_proj`.

### Sparse Block (`GptOssSparseMoeBlock`)

Combines router and experts:

1. Flatten tokens, obtain top‑k experts.
2. Permute/group tokens by expert (reuse Mixtral helpers).
3. Run expert MLPs on grouped tokens.
4. Unpermute and combine outputs with routing weights.
5. Returns optional load‑balancing metrics.

### Attention with Sink Logits (`GptOssAttention`)

Based on Levanter's attention module but augments the logits with learned per‑head *sink* values before softmax. Optionally supports layer‑specific sliding‑window masking.

### Decoder Layer and Model

Each layer performs:

1. RMSNorm → self‑attention → residual.
2. RMSNorm → sparse MoE → residual.

`GptOssTransformer` stacks layers, alternating between sliding and full attention depending on `layer_types` in the config. `GptOssLMHeadModel` wraps embeddings, the transformer, and an output head, accumulating router logits and auxiliary losses.

## Outstanding Work

* Hugging Face checkpoint conversion utilities.
* Extensive unit tests to validate parity with the PyTorch implementation.

