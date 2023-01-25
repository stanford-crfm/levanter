import dataclasses

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis
from levanter.grad_checkpointing import checkpointed_fold
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel


def test_gradient_checkpointing():
    # ensure that gradient checkpointing doesn't change the output
    # (this is a regression test for a bug that caused the output to change)
    for num_blocks in [1, 2, 4, 8, 12]:
        config = Gpt2Config(
            seq_len=16,
            hidden_dim=72,
            num_layers=num_blocks,
            num_heads=8,
            gradient_checkpointing=False,
            gradient_checkpointing_block_size=1,
        )
        config_checkpoint = dataclasses.replace(config, gradient_checkpointing=True)
        key = PRNGKey(0)

        Vocab = Axis("vocab", 128)

        model = Gpt2LMHeadModel(Vocab, config, key=key)
        model_checkpoint = Gpt2LMHeadModel(Vocab, config_checkpoint, key=key)

        input_ids = hax.arange(config.SeqLen, dtype=jnp.int32)

        a1 = model(input_ids, inference=False, key=key)
        a2 = model_checkpoint(input_ids, inference=False, key=key)

        assert hax.all(hax.isclose(a1, a2, rtol=1e-4, atol=1e-5)), f"failed with num_blocks={num_blocks}"


def test_checkpointed_fold_basic():
    def fn(carry, x):
        return carry + x

    axis = Axis("a", 4)
    assert checkpointed_fold(fn, axis, checkpoint_block_size=2)(0.0, jnp.ones(axis.size)) == 4


def test_checkpointed_fold_gradient_match():
    def fn(carry, x):
        return carry + x

    axis = Axis("a", 4)

    grad_check = jax.grad(lambda x: checkpointed_fold(fn, axis, checkpoint_block_size=2)(0.0, x))(jnp.ones(axis.size))
    grad_plain = jax.grad(lambda x: jax.lax.scan(lambda q, r: (fn(q, r), None), 0.0, x)[0])(jnp.ones(axis.size))

    assert jnp.allclose(grad_check, grad_plain)


def test_checkpointed_fold_fancier_gradient():
    def fn(carry, x):
        return jnp.log1p(jnp.exp(carry + x))

    axis = Axis("a", 32)

    grad_check = jax.grad(lambda x: checkpointed_fold(fn, axis, checkpoint_block_size=2)(0.0, x))(jnp.ones(axis.size))
    grad_plain = jax.grad(lambda x: jax.lax.scan(lambda q, r: (fn(q, r), None), 0.0, x)[0])(jnp.ones(axis.size))

    assert jnp.allclose(grad_check, grad_plain)
