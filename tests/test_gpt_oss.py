import tempfile

import chex
import jax.random as random
import pytest

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.models.gpt_oss import GptOssConfig, GptOssLMHeadModel, HfGptOssConfig
from test_utils import skip_if_no_torch
from transformers import AutoConfig, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel


AutoConfig.register("gpt_oss", HfGptOssConfig)


def _tiny_tokenizer():
    tok = Tokenizer(WordLevel({"<pad>": 0, "a": 1, "b": 2}, unk_token="<unk>"))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")


@skip_if_no_torch
def test_gpt_oss_config_roundtrip():
    hf_config = HfGptOssConfig(
        num_hidden_layers=2,
        num_local_experts=4,
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        sliding_window=32,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        output_router_logits=False,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
    )
    lev_config = GptOssConfig.from_hf_config(hf_config)
    new_hf_config = lev_config.to_hf_config(vocab_size=hf_config.vocab_size)
    for attr in [
        "num_hidden_layers",
        "num_local_experts",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "sliding_window",
        "num_experts_per_tok",
        "router_aux_loss_coef",
        "output_router_logits",
        "max_position_embeddings",
    ]:
        assert getattr(new_hf_config, attr) == getattr(hf_config, attr)


def test_gpt_oss_lm_head_model_forward():
    config = GptOssConfig(
        hidden_dim=64,
        intermediate_dim=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        seq_len=32,
        router_aux_loss_coef=None,
    )
    Batch = hax.Axis("batch", 2)
    Vocab = hax.Axis("vocab", 50)
    Pos = config.Pos
    input_ids = hax.random.randint(random.PRNGKey(0), (Batch, Pos), 0, Vocab.size)
    mask = AttentionMask.causal()
    model = GptOssLMHeadModel.init(Vocab, config, key=random.PRNGKey(1))
    out = model(input_ids, mask)
    assert out.array.shape == (Batch.size, Pos.size, Vocab.size)


@skip_if_no_torch
def test_gpt_oss_hf_roundtrip():
    tokenizer = _tiny_tokenizer()
    config = GptOssConfig(
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        seq_len=16,
        router_aux_loss_coef=None,
    )
    hf_config = HfGptOssConfig(
        num_hidden_layers=config.num_layers,
        num_local_experts=config.num_local_experts,
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.hidden_dim,
        intermediate_size=config.intermediate_dim,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads,
        sliding_window=config.sliding_window or 0,
        num_experts_per_tok=config.num_experts_per_tok,
        router_aux_loss_coef=0.0,
        output_router_logits=config.output_router_logits,
        max_position_embeddings=config.seq_len,
        rms_norm_eps=config.layer_norm_epsilon,
    )
    with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as save_dir:
        hf_config.save_pretrained(ref_dir)
        converter = config.hf_checkpoint_converter(ref_checkpoint=ref_dir, tokenizer=tokenizer)
        Vocab = hax.Axis("vocab", tokenizer.vocab_size)
        model = GptOssLMHeadModel.init(Vocab, config, key=random.PRNGKey(0))
        converter.save_pretrained(model, save_dir, save_tokenizer=False)
        loaded = converter.load_pretrained(GptOssLMHeadModel, ref=save_dir, config=config)
        chex.assert_trees_all_close(model, loaded, rtol=1e-5, atol=1e-5)
