import tempfile

import jax.numpy as jnp

import haliax as hax

from levanter.data.text import LMDatasetConfig
from levanter.models.lm_model import LmExample
from levanter.models.loss import next_token_loss


def test_dont_blow_up_without_validation_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LMDatasetConfig(
            train_urls=["kaa"],
            validation_urls=[],
            cache_dir=tmpdir,
        )

        # mostly just making sure this doesn't blow up
        assert config.validation_set(10) is None


def test_lm_example_handles_ignore_id():
    Pos = hax.Axis("Pos", 10)
    Vocab = hax.Axis("vocab", Pos.size + 1)
    tokens = hax.arange(Pos, dtype=jnp.int32)

    ignore_id = 6

    ex_ignore = LmExample.causal(tokens, ignore_id=ignore_id)
    ex_no_ignore = LmExample.causal(tokens)
    assert ex_ignore.loss_mask[Pos, ignore_id - 1] == 0

    distr = -100 * hax.nn.one_hot(ignore_id, Vocab)
    distr = distr.broadcast_axis(Pos)

    ignored_loss = next_token_loss(Pos, Vocab, distr, tokens, loss_mask=ex_ignore.loss_mask)
    no_ignore_loss = next_token_loss(Pos, Vocab, distr, tokens, loss_mask=ex_no_ignore.loss_mask)

    assert no_ignore_loss.item() >= ignored_loss.item() + 100 / Pos.size
