import tempfile

import jax.numpy as jnp
from transformers import AutoTokenizer

import haliax as hax

from levanter.data.text import BatchTokenizer, LMDatasetConfig
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from tests.test_utils import skip_if_hf_model_not_accessible


def test_dont_blow_up_without_validation_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LMDatasetConfig(
            train_urls=["kaa"],
            validation_urls=[],
            cache_dir=tmpdir,
        )

        Pos = hax.Axis("Pos", 10)
        # mostly just making sure this doesn't blow up
        assert config.validation_set(Pos) is None


def test_lm_example_handles_ignore_id():
    Pos = hax.Axis("Pos", 10)
    Vocab = hax.Axis("vocab", Pos.size + 1)
    Embed = hax.Axis("embed", 10)
    tokens = hax.arange(Pos, dtype=jnp.int32)

    ignore_id = 6
    eos_id = 10

    ex_ignore = LmExample.causal(tokens, ignore_id=ignore_id, eos_id=eos_id)
    ex_no_ignore = LmExample.causal(tokens, eos_id=eos_id)
    assert ex_ignore.loss_mask[Pos, ignore_id - 1] == 0

    logits = hax.ones((Pos, Embed))
    lm_head = hax.zeros((Embed, Vocab))
    lm_head = lm_head.at[Vocab, ignore_id].set(-100)

    ignored_loss = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, logits, lm_head, tokens, loss_mask=ex_ignore.loss_mask
    )
    no_ignore_loss = maybe_fused_next_token_loss(
        Pos, Embed, Vocab, logits, lm_head, tokens, loss_mask=ex_no_ignore.loss_mask
    )

    assert no_ignore_loss.item() >= ignored_loss.item() + 100 / Pos.size


def test_merge_split_encodings():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # make this very short for testing

    lorem = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

    short_batch_tokenizer = BatchTokenizer(tokenizer, _workaround_len=len(lorem) // 3)
    # force this
    short_batch_tokenizer._needs_long_sequence_workaround = True

    batch_tokenizer = BatchTokenizer(tokenizer, _workaround_len=50000)
    batch = [lorem]

    short_out = short_batch_tokenizer(batch)
    reg_out = batch_tokenizer(batch)

    assert short_out == reg_out


@skip_if_hf_model_not_accessible("meta-llama/Llama-2-7b-hf")
def test_llama_tokenizer_needs_long_sequence_workaround():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    batch_tokenizer = BatchTokenizer(tokenizer)
    assert batch_tokenizer._needs_long_sequence_workaround
