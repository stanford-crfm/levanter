import numpy as np
from jax.random import PRNGKey
from utils import skip_if_no_torch

import haliax as hax
from haliax.nn.attention import alibi_attention_bias, dot_product_attention_weights, forgetful_causal_mask


def test_alibi_attention_bias():
    KeySeqLen = hax.Axis("KeySeqLen", 20)
    NumHeads = hax.Axis("NumHeads", 1)
    Hid = hax.Axis("Hid", 8)

    bias = alibi_attention_bias(NumHeads, KeySeqLen)

    query = hax.ones((NumHeads, Hid))
    key = hax.ones((KeySeqLen, NumHeads, Hid))

    weights_bias = dot_product_attention_weights(Hid, KeySeqLen, query, key, bias=bias)
    weights_no_bias = dot_product_attention_weights(Hid, KeySeqLen, query, key)

    assert weights_bias.take(KeySeqLen, -1).item() > weights_bias.take(KeySeqLen, -2).item()
    assert weights_bias.take(KeySeqLen, -1).item() > weights_no_bias.take(KeySeqLen, -1).item()

    assert weights_no_bias.take(KeySeqLen, -1).item() == weights_no_bias.take(KeySeqLen, -2).item()


@skip_if_no_torch
def test_alibi_attention_compared_to_hf():
    import torch
    from transformers.models.bloom.modeling_bloom import build_alibi_tensor

    L = hax.Axis("L", 128)
    H = hax.Axis("NumHeads", 16)

    # Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
    torch_tensor = (
        build_alibi_tensor(torch.ones(1, L.size), H.size, dtype=torch.float32).numpy().reshape(H.size, L.size)
    )

    hax_tensor = np.array(alibi_attention_bias(H, L).array)

    assert np.allclose(torch_tensor, hax_tensor)


def test_fcm_attention_mask():
    KeySeqLen = hax.Axis("KeySeqLen", 20)

    mask = forgetful_causal_mask(KeySeqLen, mask_prob=0.6, sample_prob=False, key=PRNGKey(0))

    assert mask.axes == (KeySeqLen,)
    assert mask.array[0].item() == 1

    assert mask.astype(float).sum().item() <= KeySeqLen.size

    QuerySeqLen = hax.Axis("QuerySeqLen", 10)
    Head = hax.Axis("Head", 8)

    query = hax.arange(QuerySeqLen).broadcast_axis(Head)
    key = hax.arange(KeySeqLen).broadcast_axis(Head)

    weights = dot_product_attention_weights(Head, KeySeqLen, query, key, mask=mask)

    # check that all masked out values are zero
    # TODO: think about how to make this work with named arrays
    weights = weights.rearrange((KeySeqLen, QuerySeqLen)).array
    mask = mask.array

    assert weights[mask == 0].sum() == 0
    assert weights[mask == 1].sum() > 0
