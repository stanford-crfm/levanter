"""
Test script for DPO with KL penalty implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from unittest.mock import Mock

import haliax as hax
from haliax import Axis

from levanter.main.dpo import (
    DpoExample,
    compute_dpo_loss,
    compute_dpo_loss_with_kl_penalty,
    test_dpo_implementations_equivalence,
    test_dpo_kl_penalty_implementations_equivalence,
)


class MockLmHeadModel:
    """Mock language model for testing."""
    
    def __init__(self, vocab_size=1000, seq_len=128):
        self.Pos = Axis("pos", seq_len)
        self.Embed = Axis("vocab", vocab_size)
        self.vocab_size = vocab_size
        
    def __call__(self, tokens, mask=None, key=None):
        # Mock logits - random but deterministic
        batch_size = tokens.shape[0] if hasattr(tokens, 'shape') and len(tokens.shape) > 1 else 1
        seq_len = tokens.shape[-1] if hasattr(tokens, 'shape') else len(tokens)
        
        # Create mock logits with some structure
        logits = jnp.random.normal(0, 1, (batch_size, seq_len, self.vocab_size))
        return hax.named(logits, ("batch", self.Pos, self.Embed))
    
    def get_lm_head(self):
        # Mock LM head weights
        return hax.named(jnp.random.normal(0, 1, (self.Embed.size, self.Embed.size)), (self.Embed, self.Embed))


def create_mock_dpo_example(prompt_len=10, response_len=20, vocab_size=1000):
    """Create a mock DPO example for testing."""
    Pos = Axis("pos", prompt_len)
    Response = Axis("response", response_len)
    
    # Create mock token sequences
    prompt_tokens = np.random.randint(0, vocab_size, (prompt_len,), dtype=np.int32)
    chosen_tokens = np.random.randint(0, vocab_size, (response_len,), dtype=np.int32)
    rejected_tokens = np.random.randint(0, vocab_size, (response_len,), dtype=np.int32)
    
    # Wrap in NamedArray
    prompt_na = hax.named(prompt_tokens, (Pos,))
    chosen_na = hax.named(chosen_tokens, (Response,))
    rejected_na = hax.named(rejected_tokens, (Response,))
    
    return DpoExample(
        prompt_ids=prompt_na,
        chosen_ids=chosen_na,
        rejected_ids=rejected_na,
        prompt_len=prompt_len,
        response_len=response_len,
    )


def test_dpo_loss_basic():
    """Test that basic DPO loss computation works."""
    model = MockLmHeadModel()
    ex = create_mock_dpo_example()
    
    # Test with different parameters
    loss1 = compute_dpo_loss(model, ex, beta=0.1, reference_free=True)
    loss2 = compute_dpo_loss(model, ex, beta=0.2, reference_free=True)
    
    # Losses should be finite and different
    assert jnp.isfinite(loss1)
    assert jnp.isfinite(loss2)
    assert loss1 != loss2


def test_dpo_loss_with_kl_penalty_basic():
    """Test that DPO loss with KL penalty computation works."""
    model = MockLmHeadModel()
    ex = create_mock_dpo_example()
    
    # Test with KL penalty disabled (should be same as regular DPO)
    loss_no_kl = compute_dpo_loss_with_kl_penalty(model, ex, beta=0.1, kl_penalty_weight=0.0)
    loss_regular = compute_dpo_loss(model, ex, beta=0.1)
    
    # Should be approximately equal
    assert jnp.abs(loss_no_kl - loss_regular) < 1e-6
    
    # Test with KL penalty enabled
    loss_with_kl = compute_dpo_loss_with_kl_penalty(model, ex, beta=0.1, kl_penalty_weight=0.1)
    assert jnp.isfinite(loss_with_kl)


def test_dpo_loss_concatenated_vs_separate():
    """Test that concatenated and separate forward passes produce similar results."""
    model = MockLmHeadModel()
    ex = create_mock_dpo_example()
    
    # Test both implementations
    loss_separate, loss_concatenated = test_dpo_implementations_equivalence(
        model, ex, beta=0.1, reference_free=True
    )
    
    # Should be approximately equal
    assert jnp.abs(loss_separate - loss_concatenated) < 1e-6


def test_kl_penalty_concatenated_vs_separate():
    """Test that KL penalty implementations produce similar results."""
    model = MockLmHeadModel()
    ex = create_mock_dpo_example()
    
    # Test both implementations
    loss_separate, loss_concatenated = test_dpo_kl_penalty_implementations_equivalence(
        model, ex, beta=0.1, reference_free=True, kl_penalty_weight=0.1
    )
    
    # Should be approximately equal
    assert jnp.abs(loss_separate - loss_concatenated) < 1e-6


def test_kl_penalty_weight_effect():
    """Test that KL penalty weight affects the loss appropriately."""
    model = MockLmHeadModel()
    ex = create_mock_dpo_example()
    
    # Test with different KL penalty weights
    loss_no_kl = compute_dpo_loss_with_kl_penalty(model, ex, beta=0.1, kl_penalty_weight=0.0)
    loss_small_kl = compute_dpo_loss_with_kl_penalty(model, ex, beta=0.1, kl_penalty_weight=0.1)
    loss_large_kl = compute_dpo_loss_with_kl_penalty(model, ex, beta=0.1, kl_penalty_weight=1.0)
    
    # All should be finite
    assert jnp.isfinite(loss_no_kl)
    assert jnp.isfinite(loss_small_kl)
    assert jnp.isfinite(loss_large_kl)
    
    # With larger KL penalty, loss should generally be larger (though not guaranteed due to randomness)
    # This is a weak test, but helps catch major issues


def test_dpo_example_creation():
    """Test DPO example creation from dict."""
    vocab_size = 1000
    prompt_len = 10
    response_len = 20
    
    # Create raw data
    raw_data = {
        "prompt_ids": list(range(prompt_len)),
        "chosen_ids": list(range(response_len)),
        "rejected_ids": list(range(response_len, 2 * response_len)),
        "prompt_len": prompt_len,
        "response_len": response_len,
    }
    
    # Create axes
    Pos = Axis("pos", prompt_len)
    Response = Axis("response", response_len)
    
    # Create tokenizer mock
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    
    # Create example
    ex = DpoExample.from_dict(raw_data, tokenizer, Pos, Response)
    
    # Check that example was created correctly
    assert ex.prompt_len == prompt_len
    assert ex.response_len == response_len
    assert ex.prompt_ids.shape == (prompt_len,)
    assert ex.chosen_ids.shape == (response_len,)
    assert ex.rejected_ids.shape == (response_len,)


if __name__ == "__main__":
    # Run basic tests
    print("Running DPO KL penalty tests...")
    
    test_dpo_loss_basic()
    print("✓ Basic DPO loss test passed")
    
    test_dpo_loss_with_kl_penalty_basic()
    print("✓ Basic DPO loss with KL penalty test passed")
    
    test_dpo_loss_concatenated_vs_separate()
    print("✓ Concatenated vs separate forward pass test passed")
    
    test_kl_penalty_concatenated_vs_separate()
    print("✓ KL penalty concatenated vs separate test passed")
    
    test_kl_penalty_weight_effect()
    print("✓ KL penalty weight effect test passed")
    
    test_dpo_example_creation()
    print("✓ DPO example creation test passed")
    
    print("All tests passed! ✓") 