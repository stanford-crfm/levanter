# DPO with KL Divergence Penalty

This document describes the Direct Preference Optimization (DPO) implementation with optional KL divergence penalty in Levanter.

## Overview

The DPO training with KL penalty adds a regularization term that penalizes the KL divergence between the current model's output distribution and the reference model's distribution. This helps prevent the model from deviating too far from the reference during training, which can improve stability and prevent overfitting to the preference data.

## Mathematical Formulation

The total loss is computed as:

```
L_total = L_DPO + λ * L_KL
```

Where:
- `L_DPO` is the standard DPO loss
- `L_KL` is the KL divergence penalty
- `λ` is the KL penalty weight (`kl_penalty_weight`)

The KL divergence is computed as:

```
KL(p_current || p_reference) = Σ p_current * (log(p_current) - log(p_reference))
```

## Configuration

To enable KL penalty, set the `kl_penalty_weight` parameter in your configuration:

```yaml
# DPO specific configuration
beta: 0.1  # DPO temperature parameter
reference_free: False  # Must be False to use KL penalty
kl_penalty_weight: 0.1  # Weight for KL divergence penalty (0.0 disables)
```

### Key Parameters

- `beta`: DPO temperature parameter (default: 0.1)
- `kl_penalty_weight`: Weight for KL divergence penalty (default: 0.0, disabled)
- `reference_free`: Whether to use reference-free DPO (must be False for KL penalty)
- `use_concatenated_forward`: Use efficient concatenated forward passes (default: True)

## Usage

### Basic Usage

```bash
python -m levanter.main.dpo --config config/dpo_with_kl_penalty.yaml
```

### Example Configuration

See `config/dpo_with_kl_penalty.yaml` for a complete example configuration.

## Implementation Details

### Forward Pass Strategies

The implementation supports two forward pass strategies:

1. **Separate Forward Passes**: Computes chosen and rejected sequences separately
   - Simpler to understand and debug
   - Less efficient for FSDP

2. **Concatenated Forward Passes**: Concatenates sequences into a single batch
   - More efficient for FSDP
   - Reduces memory allocation/deallocation cycles

### KL Penalty Computation

The KL penalty is computed as follows:

1. Get probability distributions from current model and reference model
2. Compute KL divergence for both chosen and rejected responses
3. Average over sequence length and batch dimensions
4. Apply the penalty weight

### Reference Model Requirements

To use KL penalty, you must:

1. Set `reference_free: False`
2. Ensure your model has a `reference_model` attribute
3. The reference model should be compatible with the current model

## Benefits

1. **Stability**: Prevents the model from deviating too far from the reference
2. **Regularization**: Helps prevent overfitting to preference data
3. **Controlled Training**: Allows fine-grained control over model behavior

## Trade-offs

1. **Computational Cost**: Additional forward passes through reference model
2. **Memory Usage**: Requires storing reference model in memory
3. **Hyperparameter Tuning**: Need to tune KL penalty weight

## Testing

Run the test suite to verify the implementation:

```bash
python tests/test_dpo_kl_penalty.py
```

## Troubleshooting

### Common Issues

1. **KL penalty not working**: Ensure `reference_free: False` and model has `reference_model`
2. **High memory usage**: Consider using concatenated forward passes
3. **Training instability**: Try reducing `kl_penalty_weight`

### Debugging

Use the test functions to verify implementation correctness:

```python
from levanter.main.dpo import test_dpo_kl_penalty_implementations_equivalence

# Test that both implementations produce equivalent results
loss_separate, loss_concatenated = test_dpo_kl_penalty_implementations_equivalence(
    model, example, beta=0.1, kl_penalty_weight=0.1
)
```

## References

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [KL Divergence in Machine Learning](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) 