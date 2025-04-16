import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from transformers import AutoTokenizer

import haliax as hax

from levanter.data.text import (
    BatchTokenizer,
    ChatLmDatasetFormat,
    MultiturnChatDataset,
    SupervisedDataset,
    UrlSingleDatasetLMConfig,
    build_lm_dataset_cache,
    preprocessor_for_format,
)
from levanter.models.lm_model import LmExample
from levanter.models.loss import maybe_fused_next_token_loss
from tests.test_utils import skip_if_hf_model_not_accessible


def test_dont_blow_up_without_validation_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = UrlSingleDatasetLMConfig(
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


def test_make_sequence_mask():
    make_output_mask = SupervisedDataset._make_sequence_mask
    # Case 1: basic case with 2 segments
    segment_ids = np.array([0, 0, 1, 1, 1])
    segment_source_len = np.array([1, 2])
    expected = np.array([0, 1, 0, 0, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)

    # Case 2: all tokens are input
    segment_ids = np.array([0, 0, 1, 1])
    segment_source_len = np.array([2, 2])
    expected = np.array([0, 0, 0, 0], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)

    # Case 3: all tokens are output
    segment_ids = np.array([0, 0, 1, 1])
    segment_source_len = np.array([0, 0])
    expected = np.array([1, 1, 1, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)

    # Case 4: alternating inputs and outputs
    segment_ids = np.array([0, 0, 0, 1, 1, 1])
    segment_source_len = np.array([2, 1])
    expected = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)

    # Case 5: single segment, mixed input/output
    segment_ids = np.array([0, 0, 0, 0])
    segment_source_len = np.array([2])
    expected = np.array([0, 0, 1, 1], dtype=np.int32)
    assert_array_equal(make_output_mask(segment_ids, segment_source_len), expected)


@pytest.fixture
def dummy_chat_data():
    messages = [
        {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there, how can I help?"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "To get to the other side."},
                {"role": "assistant", "content": "No, the other side."},
            ]
        },
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "chat.jsonl"
        with path.open("w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
        yield str(path)


def test_chat_dataset_build_and_pack(dummy_chat_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = tmpdir

        tokenizer = AutoTokenizer.from_pretrained("stanford-crfm/marin-tokenizer")

        config = UrlSingleDatasetLMConfig(
            train_urls=[dummy_chat_data], format=ChatLmDatasetFormat(messages_field="messages")
        )

        processor = preprocessor_for_format(config.format, tokenizer)

        # test the processor
        source = config.get_shard_source("train")
        for doc in source.open_shard(source.shard_names[0]):
            processor([doc])

        # test the caching
        ds = build_lm_dataset_cache(cache_dir, source, config.format, tokenizer)
        ds.await_finished()
        ds_sync = ds.as_sync_dataset()
        assert len(ds_sync) == 2
        sample = next(iter(ds))

        # these are ProcessedChatDicts
        assert sample["assistant_masks"].shape == sample["input_ids"].shape
        assert 8 < sample["assistant_masks"].sum() <= 10
        # assert sample["input_ids"].shape[0] > 20
        assert (
            tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
            == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello!<|eot_id|>\n<|start_header_id|>assistant"
            "<|end_header_id|>\nHi there, how can I help?<|eot_id|>\n"
        )

        # now test packing
        Pos = hax.Axis("Pos", 100)
        packed_ds = MultiturnChatDataset(ds, Pos, max_segments_per_example=2)
        packed_ds = packed_ds.as_sync_dataset()

        assert len(packed_ds) == 1

        for ex in packed_ds:
            assert ex.tokens.axes == (Pos,)
            assert ex.loss_mask.axes == (Pos,)
            assert ex.attn_mask.segment_ids.axes == (Pos,)
