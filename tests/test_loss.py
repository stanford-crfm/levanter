# test_cross_entropy.py
import jax.random
import pytest
import jax.numpy as jnp
import haliax as hax
from haliax import NamedArray

# Import the functions from your module
# Replace 'your_module' with the actual module name where your functions are defined
from levanter.models.loss import (
    block_wise_cross_entropy_loss,
    cross_entropy_loss_and_log_normalizers,
)
from levanter.utils.jax_utils import key_iterator


@pytest.fixture
def axes():
    """
    Define and return Haliax axes for testing.
    """
    Batch = hax.Axis("batch", size=2)
    Seq = hax.Axis("seq", size=3)
    Embed = hax.Axis("embed", size=8)
    Vocab = hax.Axis("vocab", size=16)
    return Batch, Seq, Embed, Vocab


@pytest.fixture
def test_data(axes):
    """
    Create synthetic test data for cross-entropy loss computation.
    """
    Batch, Seq, Embed, Vocab = axes

    key = key_iterator(jax.random.PRNGKey(0))

    # Initialize pred_embeddings with ones
    pred_embeddings = hax.random.normal(next(key), (Batch, Seq, Embed))

    # Initialize pred_lm_head with ones
    pred_lm_head = hax.random.normal(next(key), (Embed, Vocab))

    # Define true_ids such that the target is always the first token in vocab
    true_ids = hax.random.randint(next(key), (Batch, Seq), 0, Vocab.size)

    return pred_embeddings, pred_lm_head, true_ids


def test_basic_equivalence(axes, test_data):
    """
    Test that block-wise loss equals full loss when block_size perfectly divides vocab_size.
    """
    Batch, Seq, Embed, Vocab = axes
    pred_embeddings, pred_lm_head, true_ids = test_data

    # Compute full loss
    logits_full = hax.dot(pred_embeddings, pred_lm_head, axis="embed")
    target_y_full = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
    loss_full, norm_full = cross_entropy_loss_and_log_normalizers(logits_full, Vocab, target_y_full)

    loss_block, norm_this = block_wise_cross_entropy_loss(
        pred_embeddings=pred_embeddings,
        pred_lm_head=pred_lm_head,
        Contract=Embed,
        Label=Vocab,
        labels_y=true_ids,
        block_size=2,
    )

    # Assert that the losses are close
    assert hax.all(hax.isclose(loss_full, loss_block, atol=1e-6)), "Block-wise loss does not match full loss."


def test_single_block(axes, test_data):
    """
    Test behavior when vocab_size equals block_size.
    """
    Batch, Seq, Embed, Vocab = axes
    pred_embeddings, pred_lm_head, true_ids = test_data

    # Compute full loss
    loss_full, sumexp_full = _compute_full(Vocab, pred_embeddings, pred_lm_head, true_ids)

    # Compute block-wise loss with block_size=4 (vocab_size=4)
    with jax.disable_jit():
        loss_block, sumexp_block = block_wise_cross_entropy_loss(
            pred_embeddings=pred_embeddings,
            pred_lm_head=pred_lm_head,
            Contract=Embed,
            Label=Vocab,
            labels_y=true_ids,
            block_size=Vocab.size,
        )

    # Assert that the losses are close
    assert hax.all(hax.isclose(sumexp_full, sumexp_block, atol=1e-6)), "Single block-wise sumexp does not match full sumexp."
    assert hax.all(hax.isclose(loss_full, loss_block, atol=1e-6)), "Single block-wise loss does not match full loss."


def _compute_full(Vocab, pred_embeddings, pred_lm_head, true_ids):
    logits_full = hax.dot(pred_embeddings, pred_lm_head, axis="embed")
    target_y_full = hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype)
    loss_full, sumexp_full = cross_entropy_loss_and_log_normalizers(logits_full, Vocab, target_y_full)
    return loss_full, sumexp_full


def test_multiple_blocks(axes, test_data):
    """
    Test block-wise loss with multiple blocks.
    """
    Batch, Seq, Embed, Vocab = axes
    pred_embeddings, pred_lm_head, true_ids = test_data

    # Compute full loss
    loss_full, logz_full = _compute_full(Vocab, pred_embeddings, pred_lm_head, true_ids)

    # Compute block-wise loss with block_size=1 (vocab_size=4)
    loss_block, logz_block = block_wise_cross_entropy_loss(
        pred_embeddings=pred_embeddings,
        pred_lm_head=pred_lm_head,
        Contract=Embed,
        Label=Vocab,
        labels_y=true_ids,
        block_size=1,
    )

    # Assert that the losses are close
    assert hax.all(hax.isclose(logz_full, logz_block, atol=1e-6)), "Multiple block-wise logz does not match full logz."
    assert hax.all(hax.isclose(loss_full, loss_block, atol=1e-6)), "Multiple block-wise loss does not match full loss."



def test_block_size_not_dividing_vocab(axes, test_data):
    """
    Test that a ValueError is raised when block_size does not divide vocab_size.
    """
    Batch, Seq, Embed, Vocab = axes
    pred_embeddings, pred_lm_head, true_ids = test_data

    with pytest.raises(ValueError, match="Vocab size must be a multiple of block size"):
        block_wise_cross_entropy_loss(
            pred_embeddings=pred_embeddings,
            pred_lm_head=pred_lm_head,
            Contract=Embed,
            Label=Vocab,
            labels_y=true_ids,
            block_size=3,  # 4 is not divisible by 3
        )


def test_vocab_size_less_than_block_size(axes, test_data):
    """
    Test behavior when vocab_size is less than block_size.
    """
    Batch, Seq, Embed, Vocab = axes
    pred_embeddings, pred_lm_head, true_ids = test_data

    # Set block_size greater than vocab_size
    block_size = 5  # vocab_size=4

    with pytest.raises(ValueError, match="Vocab size must be a multiple of block size"):
        block_wise_cross_entropy_loss(
            pred_embeddings=pred_embeddings,
            pred_lm_head=pred_lm_head,
            Contract=Embed,
            Label=Vocab,
            labels_y=true_ids,
            block_size=block_size,
        )


def test_large_vocab(axes):
    """
    Test block-wise loss with a larger vocabulary.
    """
    Batch = hax.Axis("batch", size=4)
    Seq = hax.Axis("seq", size=5)
    Embed = hax.Axis("embed", size=6)
    Vocab = hax.Axis("vocab", size=12)

    pred_embeddings = NamedArray(
        jnp.ones((Batch.size, Seq.size, Embed.size)),
        axes=(Batch, Seq, Embed),
    )
    pred_lm_head = NamedArray(
        jnp.ones((Embed.size, Vocab.size)),
        axes=(Embed, Vocab),
    )
    true_ids = NamedArray(
        jnp.zeros((Batch.size, Seq.size), dtype=jnp.int32),
        axes=(Batch, Seq),
    )

    # Compute full loss
    loss_full, logz_full = cross_entropy_loss_and_log_normalizers(
        pred_y=hax.dot(pred_embeddings, pred_lm_head, axis="embed"),
        Label=Vocab,
        target_y=hax.nn.one_hot(true_ids, Vocab, dtype=pred_embeddings.dtype),
    )

    # Compute block-wise loss with block_size=3 (vocab_size=12 is divisible by 3)
    loss_block, logz_block = block_wise_cross_entropy_loss(
        pred_embeddings=pred_embeddings,
        pred_lm_head=pred_lm_head,
        Contract=Embed,
        Label=Vocab,
        labels_y=true_ids,
        block_size=3,
    )

    # Assert that the losses are close
    assert hax.all(hax.isclose(loss_full, loss_block, atol=1e-6)), "Large vocab block-wise loss does not match full loss."
    assert hax.all(hax.isclose(logz_full, logz_block, atol=1e-6)), "Large vocab block-wise logz does not match full logz."

