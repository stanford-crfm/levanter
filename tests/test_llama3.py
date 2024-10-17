import json
import tempfile

import numpy as np
from jax import random

import haliax as hax

from levanter.models.attention import AttentionMask
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from test_utils import skip_if_no_torch


def get_config(vocab_size=1000):
    from transformers import LlamaConfig

    llama3_cfg = json.loads(
        """
        {
            "architectures": [
                "LlamaForCausalLM"
            ],
            "attention_bias": false,
            "attention_dropout": 0,
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "pretraining_tp": 1,
            "rms_norm_eps": 0.00001,
            "rope_scaling": {
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
              },
            "rope_theta": 500000,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.40.0.dev0",
            "use_cache": true,
            "vocab_size": 128256
            }
    """
    )
    llama3_8b_config: LlamaConfig = LlamaConfig.from_dict(llama3_cfg)
    llama3_8b_config.hidden_size = 16
    llama3_8b_config.intermediate_size = 64
    llama3_8b_config.num_attention_heads = 4
    llama3_8b_config.num_hidden_layers = 4
    llama3_8b_config.num_key_value_heads = 2
    llama3_8b_config.max_position_embeddings = 128
    llama3_8b_config.vocab_size = vocab_size
    return llama3_8b_config


@skip_if_no_torch
def test_llama_roundtrip():
    import torch
    from transformers import AutoModelForCausalLM, LlamaForCausalLM

    Vocab = hax.Axis("vocab", 1000)
    hf_config = get_config(Vocab.size)

    converter = LlamaConfig().hf_checkpoint_converter()

    config = LlamaConfig.from_hf_config(hf_config)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = LlamaForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    # torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            LlamaLMHeadModel, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        @hax.named_jit
        def compute(model, input):
            model_output = model(input, attn_mask=attn_mask)
            return model_output

        jax_out = compute(model, input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        # now we're going to magnify the model parameters enough that differences should actualy show up
        jax_out = compute(model, input).array

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = AutoModelForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        np.testing.assert_allclose(torch_out2, jax_out, rtol=1e-5, atol=1e-5)


@skip_if_no_torch
def test_llama3_rotary_embedding():
    import torch
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding

    llama_config = get_config()
    key = random.PRNGKey(0)
    device = "cpu"

    lev_config = LlamaConfig.from_hf_config(llama_config)
    HeadSize = lev_config.HeadSize
    Pos = lev_config.Pos
    seq_len = Pos.size

    x = random.normal(key, (1, seq_len))
    x_torch = torch.from_numpy(np.array(x))

    levanter_emb = lev_config.rope.build(HeadSize, Pos)
    levanter_output = (levanter_emb.cos, levanter_emb.sin)

    hf_rope = HFLlamaRotaryEmbedding(max_position_embeddings=seq_len, device=device, config=llama_config)
    hf_output = hf_rope(x_torch, torch.arange(seq_len).reshape(1, -1))

    for jax_out, torch_out in zip(levanter_output, hf_output):
        torch_out = torch_out.numpy()
        assert np.isclose(torch_out, np.array(jax_out.array), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"
