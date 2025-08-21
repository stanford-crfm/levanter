# GPT-OSS in Levanter

This document captures the design and implementation notes for porting Hugging Face's `hf_gpt_oss.py` (a PyTorch model) to the Levanter/JAX ecosystem.

## Overview

GPT-OSS is a decoder-only transformer that combines rotary-position attention with a sparse Mixture-of-Experts (MoE) MLP. The implementation uses pure Haliax named tensor operations throughout, ensuring consistency with Levanter's design philosophy:

* **Pure Haliax operations** for all tensor manipulations, including MoE routing
* **Named axes** via [Haliax](https://github.com/stanford-crfm/haliax) for shape safety and readability
* **Equinox modules** to define parameterized layers
* **Functional JAX style** with explicit PRNG keys
* **Automatic partitioning** through Haliax's named operations

## Components

### Configuration (`GptOssConfig`)

Extends `MistralConfig` and adds MoE and routing options:

* `num_local_experts` – number of experts per MoE layer.
* `num_experts_per_tok` – how many experts a token is routed to.
* `sliding_window` – if set, some layers use sliding‑window attention.
* `router_aux_loss_coef` – weight for the load‑balancing loss.
* Axis accessors: `Experts`, `TopExperts`, etc.

### Router Implementation

The router uses a linear projection over the embedding dimension to score each expert. The top-k experts and their normalized routing weights are selected per token using Haliax's `top_k` function, which provides named tensor operations for expert selection.

### Experts (`GptOssExperts`)

Implements the gated GLU variant from the PyTorch model:

1. `gate_up_proj` → split into `gate` and `up` parts.
2. Clamp activations to `[-limit, limit]` (limit = 7.0).
3. Apply gated GELU (`gate * sigmoid(alpha * gate)`, alpha = 1.702).
4. Multiply by `(up + 1)` and project down with `down_proj`.

### Sparse Block (`GptOssSparseMoeBlock`)

Combines router and experts using pure Haliax operations:

1. **Routing**: Use `hax.top_k()` to select top-k experts per token with proper axis management
2. **Permutation**: Use `hax.argsort()`, `hax.take()`, and `hax.bincount()` for token-to-expert grouping
3. **Expert Processing**: Run expert MLPs on grouped tokens with automatic batching
4. **Unpermutation**: Use `hax.argsort()`, `hax.take()`, and `hax.unflatten_axis()` to restore token order
5. **Combination**: Combine expert outputs with routing weights using named tensor operations
6. Returns load-balancing metrics for training stability

### Attention with Sink Logits (`GptOssAttention`)

Based on Levanter's attention module but augments the logits with learned per‑head *sink* values before softmax. Optionally supports layer‑specific sliding‑window masking.

### Decoder Layer and Model

Each layer performs:

1. RMSNorm → self‑attention → residual.
2. RMSNorm → sparse MoE → residual.

`GptOssTransformer` stacks layers, alternating between sliding and full attention depending on `layer_types` in the config. `GptOssLMHeadModel` wraps embeddings, the transformer, and an output head, accumulating router logits and auxiliary losses.

## Implementation Highlights

### Pure Haliax Architecture

The GPT-OSS implementation demonstrates best practices for using Haliax named tensors:

* **Expert Routing**: All routing operations use named axes (`Experts`, `TopExperts`, `Token`) for clarity
* **Shape Safety**: Named operations prevent dimension mismatches common in MoE implementations  
* **Readability**: Code like `hax.top_k(router_probs, axis=Experts)` is self-documenting
* **Composability**: Pure Haliax operations integrate seamlessly with Levanter's distributed training

### Key Functions Used

* `hax.top_k()`: Expert selection with automatic axis management
* `hax.argsort()`: Token ordering for expert grouping
* `hax.take()`: Token selection and reordering operations
* `hax.bincount()`: Expert load counting for balancing
* `hax.unflatten_axis()`: Tensor reshaping with named dimensions

### Benefits of the Haliax Approach

1. **Debugging**: Named tensors make MoE routing logic transparent
2. **Maintenance**: Self-documenting operations reduce cognitive load
3. **Correctness**: Axis-aware operations prevent common tensor manipulation errors
4. **Performance**: Haliax operations compile to efficient JAX code with automatic partitioning

## OOM: Out of Memory Analysis

### The Error

When training with `gpt_oss_small_fast.yaml`, the following Out-of-Memory error occurred:

```
jaxlib._jax.XlaRuntimeError: RESOURCE_EXHAUSTED: Allocation (size=38654705664) would exceed memory (size=34359738368) :: #allocation598 [shape = 'bf16[12,524288,3072]{2,1,0:T(8,128)(2,1)}', space=hbm, size = 0xffffffffffffffff, tag = 'output of custom-call.400@{}'] :: <no-hlo-instruction>
```

### Step-by-Step Analysis

#### 1. Memory Numbers Breakdown
- **Attempted allocation**: 38,654,705,664 bytes = **38.6 GB**
- **Available memory**: 34,359,738,368 bytes = **32.0 GB** 
- **Memory deficit**: 4.3 GB over the limit
- **Data type**: `bf16` (bfloat16) = 2 bytes per element

#### 2. Tensor Shape Analysis
The problematic tensor has shape: `bf16[12,524288,3072]`

**Dimension breakdown**:
- **12**: Number of layers (`num_layers: 12`)
- **524,288**: Token dimension (batch × sequence)  
- **3,072**: Hidden dimension (`intermediate_dim: 3072`)

**Memory calculation**:
```
12 × 524,288 × 3,072 × 2 bytes (bf16) = 38,654,705,664 bytes = 38.6 GB
```

#### 3. The Critical Discovery: Effective Batch Size

**Expected**: `train_batch_size: 64` × `seq_len: 4096` = 262,144 tokens
**Actual**: 524,288 tokens = **2× larger than expected**

This means the effective batch size is **128**, not 64. The doubling could be due to:
- **Gradient accumulation**: Levanter may be accumulating gradients over 2 microbatches
- **Forward/backward pass**: Both activations and gradients held simultaneously
- **FSDP behavior**: Additional memory overhead from sharding strategy

#### 4. Configuration Problems Identified

**Problem 1: Excessive Sequence Length**
```yaml
# gpt_oss_small_fast.yaml (PROBLEMATIC)
seq_len: 4096                    # 4× longer than working configs

# gpt2_small_fast.yaml (WORKING)  
seq_len: 1024                    # Memory-efficient baseline
```

**Problem 2: No Gradient Checkpointing**
```yaml
# gpt_oss_small_fast.yaml (PROBLEMATIC)
gradient_checkpointing: false    # Stores all activations

# gpt2_small_fast.yaml (WORKING)
gradient_checkpointing: true     # Trades compute for memory
```

**Problem 3: Larger Model Architecture**
```yaml
# gpt_oss_small_fast.yaml
intermediate_dim: 3072           # MoE intermediate dimension
num_local_experts: 4             # Additional expert parameters

# gpt2_small_fast.yaml  
# No MoE overhead - simpler FFN layers
```

#### 5. Memory Scaling Laws for MoE Models

**Base Transformer Memory**: `O(batch_size × seq_len × hidden_dim × num_layers)`
**MoE Additional Memory**:
- **Expert parameters**: `num_experts × intermediate_dim × hidden_dim`
- **Routing logits**: `batch_size × seq_len × num_experts`  
- **Expert assignment**: Additional indexing and permutation tensors

**Memory amplification factors**:
- **4× sequence length**: 4× memory (linear scaling)
- **No gradient checkpointing**: 2-3× memory overhead  
- **MoE routing**: 1.2-1.5× memory overhead
- **Combined effect**: ~12× memory increase vs working config

#### 6. Why This Specific Allocation Failed

The error occurs during a `custom-call.400` operation, likely:
- **Expert processing**: Grouping tokens by expert assignment
- **Attention computation**: Large attention matrices for long sequences
- **Gradient computation**: Backward pass requiring activation storage

The shape `[12,524288,3072]` suggests this is intermediate MLP activations across all layers, stored simultaneously due to lack of gradient checkpointing.

### Root Cause Summary

The OOM error is caused by a **perfect storm** of memory-intensive configurations:

1. **4× sequence length** (4096 vs 1024) 
2. **No gradient checkpointing** (stores all activations)
3. **2× effective batch size** (128 vs expected 64)
4. **MoE architecture overhead** (experts + routing)
5. **Running on constrained hardware** (32GB memory limit)

### Proposed Fixes

**Immediate fixes** (in order of impact):

1. **Enable gradient checkpointing**:
   ```yaml
   model:
     gradient_checkpointing: true    # 2-3× memory reduction
   ```

2. **Reduce sequence length**:
   ```yaml
   model:
     seq_len: 1024                   # 4× memory reduction  
   ```

3. **Reduce batch size**:
   ```yaml
   trainer:
     train_batch_size: 32            # 2× memory reduction
   ```

4. **Investigation needed**:
   - Why effective batch size is 128 instead of 64
   - Whether FSDP settings are optimal for this model size
   - If expert routing is creating additional memory overhead

**Conservative "small_fast" configuration**:
```yaml
model:
  seq_len: 1024                     # Start small, scale up
  gradient_checkpointing: true      # Essential for memory efficiency
  
trainer:
  train_batch_size: 32              # Conservative batch size
```

### Memory Estimation Guidelines

For future GPT-OSS configurations, estimate memory as:

```
Base Memory = batch_size × seq_len × hidden_dim × num_layers × 2 (bf16)
MoE Overhead = Base Memory × 0.3 (expert parameters + routing)
No Grad Checkpoint = Total × 2.5 (activation storage)
Safety Margin = Total × 1.2 (JAX/XLA overhead)

Total Estimate = (Base Memory × 1.3 × 2.5 × 1.2) if no grad checkpointing
               = (Base Memory × 1.3 × 1.2) if grad checkpointing enabled
```

**Example for fixed config**:
```
Base = 32 × 1024 × 768 × 12 × 2 = 603 MB
With MoE = 603 MB × 1.3 = 784 MB  
With Grad Checkpoint = 784 MB × 1.2 = 941 MB
```

This fits comfortably in 32GB memory with room for optimizer states and other overhead.

### Follow-up Analysis: Massive OOM Even with 1024 Sequence Length

**Second OOM Error**: Even after reducing `seq_len` to 1024, encountered **98.33 GB allocation** on 30.75 GB device.

**Critical Finding**: **10 identical 9GB allocations** of shape `bf16[12,131072,3072]`
- Each allocation: 12 × 131,072 × 3,072 × 2 bytes = 9GB
- **All layer activations stored simultaneously** - this is the smoking gun!

**Root Cause**: `gradient_checkpointing: false` in config means:
- Every transformer layer's MLP activations stored in memory 
- No memory recomputation during backward pass
- 10+ layers × 9GB = 90GB+ just for activations

### How Gradient Checkpointing Works in Levanter

**Implementation Location**: `/haliax/src/haliax/nn/scan.py`

**Key Classes**:
1. **`Stacked`**: Used for layer stacking with gradient checkpointing
2. **`BlockSeq`**: Alternative layer implementation 
3. **`ScanCheckpointPolicy`**: Controls checkpointing behavior

**In GPT-OSS**:
```python
# src/levanter/models/gpt_oss.py:340
layers = S.init(config.Layers, GptOssDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
    config, key=shaped_rng_split(key, config.num_layers)
)
```

**Config Inheritance**:
- `GptOssConfig` inherits from `MistralConfig`
- `MistralConfig` defines: `gradient_checkpointing: bool = True` (default enabled)
- But can be overridden in YAML config

**How It Works**:
1. **Without checkpointing** (`false`): All layer activations stored throughout forward pass
2. **With checkpointing** (`true`): Only current layer activations stored, others recomputed in backward pass
3. **Memory tradeoff**: More compute for less memory (typically 2-3× memory reduction)

**YAML Configuration**:
```yaml
model:
  gradient_checkpointing: true   # Enable memory-efficient training
  # OR
  gradient_checkpointing: false  # Store all activations (memory intensive)
```

**Why This Wasn't Working**: 
The `gpt_oss_small_fast.yaml` explicitly set `gradient_checkpointing: false`, overriding the safe default, causing the massive memory explosion.

### Follow-up Issue: TPU v4-128 Parallelism Configuration

**Third Error**: After enabling gradient checkpointing, encountered `ZeroDivisionError: integer division or modulo by zero` in trainer validation.

**Root Cause**: Missing TPU parallelism settings caused `data_axis_size` to be 0.

**Error Location**: 
```python
# src/levanter/trainer.py:949
elif self.train_batch_size % (self.per_device_parallelism * self.data_axis_size) != 0:
```

**Missing Settings**:
- `model_axis_size`: Controls tensor parallelism (default should be 1 for small models)
- `per_device_parallelism`: Controls data parallelism (use -1 for auto-calculation)

**TPU v4-128 Requirements**:
- **128 devices total**: Each device needs adequate batch size
- **Minimum batch size**: Likely 128 (1 per device) for efficient utilization
- **Data axis calculation**: 
  ```
  data_axis_size = data_dcn_axis_size * data_ici_axis_size * replica_dcn_axis_size * replica_ici_axis_size
  ```

**Fixed Configuration**:
```yaml
trainer:
  model_axis_size: 1               # No tensor parallelism for small model
  per_device_parallelism: -1       # Auto-calculate
  train_batch_size: 128            # 1 per device on v4-128
```

**Expected Behavior**: 
- `per_device_parallelism` auto-calculated as: `128 batch_size / 128 devices = 1`
- Each TPU device processes 1 sample per step
- With gradient checkpointing, memory per device should be manageable

## Complete Configuration Issues Summary

The GPT-OSS "small fast" configuration encountered a **chain of three critical issues** that had to be resolved sequentially:

### Timeline of Issues and Fixes

#### Issue #1: Original Massive OOM (98GB+ allocation)
**Symptom**: `RESOURCE_EXHAUSTED: Allocation (size=38654705664) would exceed memory (size=34359738368)`
- **Error Shape**: `bf16[12,524288,3072]` = 38.6GB
- **Root Cause**: `gradient_checkpointing: false` + `seq_len: 4096` + `batch_size: 64` (effective 128)
- **Impact**: 10+ identical 9GB allocations storing all layer activations simultaneously
- **Fix Applied**: 
  ```yaml
  model:
    gradient_checkpointing: true  # Enable memory-efficient training
    seq_len: 1024                # Reduce from 4096
  ```

#### Issue #2: Persistent OOM Even with Fixes
**Symptom**: `Used 98.33G of 30.75G hbm. Exceeded hbm capacity by 67.59G`
- **Error**: 10 identical allocations of `bf16[12,131072,3072]` = 9GB each
- **Root Cause**: Still had `gradient_checkpointing: false` in configuration
- **Critical Discovery**: Each layer's MLP activations (9GB) stored simultaneously
- **Fix Applied**: Correctly enabled `gradient_checkpointing: true`

#### Issue #3: TPU v4-128 Parallelism Error  
**Symptom**: `ZeroDivisionError: integer division or modulo by zero`
- **Error Location**: `trainer.py:949` in batch size validation
- **Root Cause**: Missing TPU parallelism configuration caused `data_axis_size = 0`
- **Missing Settings**: `model_axis_size` and `per_device_parallelism`
- **Fix Applied**:
  ```yaml
  trainer:
    model_axis_size: 1           # No tensor parallelism
    per_device_parallelism: -1   # Auto-calculate
    train_batch_size: 128        # Minimum for TPU v4-128
  ```

### Technical Deep Dive

#### Memory Explosion Mechanism
```
Without Gradient Checkpointing:
Layer 0 MLP: 12 × 131072 × 3072 × 2 bytes = 9GB stored
Layer 1 MLP: 12 × 131072 × 3072 × 2 bytes = 9GB stored  
Layer 2 MLP: 12 × 131072 × 3072 × 2 bytes = 9GB stored
...
Layer 11 MLP: 12 × 131072 × 3072 × 2 bytes = 9GB stored
Total: 10+ × 9GB = 90GB+ for just MLP activations

With Gradient Checkpointing:
Only current layer stored: 9GB maximum
Total model memory: ~10-15GB (manageable)
```

#### TPU v4-128 Architecture Requirements
```
Total devices: 128
Per-device memory: ~30GB HBM
Minimum batch size: 128 (1 sample per device)
Data parallelism: batch_size / num_devices = 128 / 128 = 1

data_axis_size calculation:
= data_dcn_axis_size × data_ici_axis_size × replica_dcn_axis_size × replica_ici_axis_size
= Must be > 0 for validation to pass
```

#### MoE-Specific Memory Overhead
```
Base Transformer: batch_size × seq_len × hidden_dim × num_layers
MoE Additional:
- Expert parameters: num_experts × intermediate_dim × hidden_dim  
- Routing logits: batch_size × seq_len × num_experts
- Token permutation: Additional indexing and gathering operations
- Estimated overhead: +30% memory vs standard transformer
```

### Final Working Configuration

```yaml
# Memory-efficient GPT-OSS configuration for TPU v4-128
model:
  type: gpt_oss
  seq_len: 1024                    # Conservative sequence length
  hidden_dim: 768
  intermediate_dim: 3072
  num_layers: 12
  num_heads: 12
  num_kv_heads: 4
  num_local_experts: 4
  num_experts_per_tok: 2
  gradient_checkpointing: true     # CRITICAL: Enable for memory efficiency

trainer:
  model_axis_size: 1               # No tensor parallelism for small model
  per_device_parallelism: -1       # Auto-calculate based on batch size
  train_batch_size: 128            # Minimum for TPU v4-128 (1 per device)
  mp: p=f32,c=bfloat16            # Mixed precision training
```

### Memory Validation (Fixed Configuration)

```
Per-device calculation:
- Devices: 128 TPU v4 chips  
- Batch per device: 128 ÷ 128 = 1 sample
- Sequence length: 1024 tokens
- Hidden size: 768, Intermediate: 3072

Memory per device:
- Current layer activations: 1 × 1024 × 3072 × 2 bytes = 6MB
- Model parameters: ~100MB (distributed via FSDP)
- Optimizer state: ~200MB
- Total estimated: <1GB per device

Available per device: 30GB
Safety margin: 30× overhead available
```

### Lessons Learned

1. **Gradient Checkpointing is Essential**: For models with >8 layers, gradient checkpointing can reduce memory by 5-10×
2. **TPU Configuration Complexity**: Missing parallelism settings can cause cryptic division-by-zero errors
3. **MoE Memory Patterns**: Expert routing creates additional memory pressure beyond standard transformers
4. **Sequential Debugging**: Configuration issues often mask each other - must fix systematically
5. **Hardware-Specific Requirements**: TPU v4-128 has minimum batch size requirements that affect configuration

### Configuration Checklist for Future GPT-OSS Deployments

✅ **Memory Settings**:
- `gradient_checkpointing: true` (unless you have >100GB memory per device)
- Conservative `seq_len` (start with 1024, scale up)
- Reasonable `train_batch_size` for your hardware

✅ **TPU Settings** (for v4-128):
- `model_axis_size: 1` (for models <7B parameters)
- `per_device_parallelism: -1` (auto-calculate)
- `train_batch_size: 128` (minimum for efficient utilization)

✅ **MoE Settings**:
- Monitor `router_aux_loss_coef` for training stability
- Consider `num_experts_per_tok = 2` as safe default
- Watch for routing entropy in metrics

## Outstanding Work

* Enhanced Hugging Face checkpoint conversion utilities
* Performance benchmarking against reference implementations  
* Memory optimization testing with various sequence lengths
* TPU v4-128 scaling experiments with larger models
* Investigation of the 2× batch size multiplier phenomenon

