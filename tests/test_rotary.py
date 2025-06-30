import numpy as np
import pytest
from jax import random

import haliax as hax

from levanter.layers.rotary import _rotate_half as levanter_rotate_half
from test_llama import _get_llama_config
from test_utils import skip_if_no_torch


@skip_if_no_torch
@pytest.mark.parametrize("test_seq_len", [64, 128, 256])
def test_apply_rotary_pos_emb(test_seq_len):
    import torch
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary_pos_emb
    from transformers.models.llama.modeling_llama import rotate_half as hf_rotate_half

    def assert_equal_out(hax_out, torch_out: torch.Tensor):
        assert np.isclose(
            torch_out.numpy(), np.array(hax_out.array), rtol=1e-2, atol=1e-2
        ).all(), f"{torch_out} != {hax_out}"

    def named_array_to_tensor(named_array):
        return torch.from_numpy(np.array(named_array.array))

    _llama_config = _get_llama_config()
    Pos = _llama_config.Pos
    Heads = _llama_config.attention_config().Heads
    HeadSize = _llama_config.attention_config().HeadSize
    Batch = hax.Axis("batch", 3)

    # note here we switch Heads and Pos for the shape of the output tensors
    q = hax.random.normal(random.PRNGKey(0), (Batch, Pos, Heads, HeadSize))
    k = hax.random.normal(random.PRNGKey(1), (Batch, Pos, Heads, HeadSize))

    # Check the output of _rotate_half() from levanter and hf
    levanter_out_rf_q = levanter_rotate_half(q, HeadSize)
    levanter_out_rf_k = levanter_rotate_half(k, HeadSize)

    q_tensor = named_array_to_tensor(q).transpose(1, 2)  # needed for HF
    k_tensor = named_array_to_tensor(k).transpose(1, 2)
    hf_out_rf_q = hf_rotate_half(q_tensor).transpose(1, 2)  # re-transpose to match levanter
    hf_out_rf_k = hf_rotate_half(k_tensor).transpose(1, 2)

    assert_equal_out(levanter_out_rf_q, hf_out_rf_q)
    assert_equal_out(levanter_out_rf_k, hf_out_rf_k)

    rot_config = _llama_config.rope
    rot = rot_config.build(HeadSize)
    hf_rot = LlamaRotaryEmbedding(_llama_config.to_hf_config(1000))

    position_ids = hax.arange(Pos)

    lev_rot_q = rot(q, position_ids)
    lev_rot_k = rot(k, position_ids)

    # actually apply the rotary embeddings
    hf_cos, hf_sin = hf_rot(q_tensor, named_array_to_tensor(position_ids).reshape(1, -1))
    hf_rot_q, hf_rot_k = hf_apply_rotary_pos_emb(q_tensor, k_tensor, hf_cos[:, :], hf_sin[:, :])
    hf_rot_q = hf_rot_q.transpose(1, 2)
    hf_rot_k = hf_rot_k.transpose(1, 2)

    assert_equal_out(lev_rot_q, hf_rot_q)
    assert_equal_out(lev_rot_k, hf_rot_k)
