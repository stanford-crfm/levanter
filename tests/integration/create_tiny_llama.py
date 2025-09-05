#!/usr/bin/env python3
"""
Create a tiny Llama model for testing purposes.

This script creates a very small Llama model, saves it in Hugging Face format,
and then loads it back to verify it works. This gives us a fast, reliable
test model for integration tests.
"""

import tempfile
import shutil
from pathlib import Path

import jax.random as random
import haliax as hax
from transformers import LlamaForCausalLM

from levanter.models.llama import LlamaConfig, LlamaLMHeadModel


def create_tiny_llama_model():
    """Create a tiny Llama model for testing."""

    # Create a very small config
    config = LlamaConfig(
        seq_len=64,           # Very short sequence length
        hidden_dim=32,        # Tiny hidden dimension
        intermediate_dim=16,  # Tiny intermediate dimension
        num_heads=2,          # Minimal number of heads
        num_kv_heads=1,       # Single KV head
        num_layers=2,         # Just 2 layers
        gradient_checkpointing=False,
        scan_layers=False,
    )

    print("Creating tiny Llama model with config:")
    print(f"  seq_len: {config.seq_len}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  intermediate_dim: {config.intermediate_dim}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_kv_heads: {config.num_kv_heads}")
    print(f"  num_layers: {config.num_layers}")

    # Create vocab axis
    Vocab = hax.Axis("vocab", 1000)  # Small vocabulary

    # Initialize the model
    key = random.PRNGKey(42)
    model = LlamaLMHeadModel.init(Vocab=Vocab, config=config, key=key)

    print(" Model created successfully")
    print(f"  Total parameters: {sum(x.size for x in hax.tree_util.tree_flatten(model)[0])}")

    return model, config, Vocab


def save_as_hf_model(model, config, vocab_size, output_dir: str):
    """Save the Levanter model as a Hugging Face model."""

    print(f"Saving model to {output_dir}")

    # Convert config to HF format
    hf_config = config.to_hf_config(vocab_size)

    # Create a temporary directory for the HF model
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the Levanter model in HF format
        # We'll use the config's hf_checkpoint_converter method
        converter = config.hf_checkpoint_converter()
        converter.save_pretrained(model, tmpdir, save_reference_code=False)

        # Load it back as an HF model to verify it works
        print("Loading saved model to verify...")
        hf_model = LlamaForCausalLM.from_pretrained(tmpdir)

        # Save to final output directory
        hf_model.save_pretrained(output_dir)
        hf_config.save_pretrained(output_dir)

        print(f"Model saved successfully to {output_dir}")

def create_test_tokenizer(output_dir: str):
    """Create a simple test tokenizer."""
    from transformers import LlamaTokenizer

    # Create a minimal tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

    # We can't set vocab_size directly, but we can create a simple tokenizer
    # The model will handle the vocabulary size internally
    tokenizer.model_max_length = 64

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f" Tokenizer saved to {output_dir}")

    # Print some info about the tokenizer
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"   Model max length: {tokenizer.model_max_length}")


def main():
    """Main function to create and save the tiny model."""

    # Output directory for the test model
    output_dir = Path("tests/integration/tiny_llama_test_model")

    # Clean up existing directory
    if output_dir.exists():
        print(f"🧹 Cleaning up existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create the tiny model
        model, config, vocab = create_tiny_llama_model()

        # Save as HF model
        save_as_hf_model(model, config, vocab.size, str(output_dir))

        # Create test tokenizer
        create_test_tokenizer(str(output_dir))

        print("Tiny Llama test model created successfully!")
        print(f"Model saved to: {output_dir}")
        print(" You can now use this model in integration tests with:")
        print(f"  --checkpoint_path {output_dir}")
        print(f"  --tokenizer {output_dir}")

    except Exception as e:
        print(f"Error creating test model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
