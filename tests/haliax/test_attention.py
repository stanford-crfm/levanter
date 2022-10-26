import haliax as hax
from haliax.nn.attention import alibi_attention_bias, dot_product_attention_weights


def test_alibi_attention_bias():
    KeySeqLen = hax.Axis("KeySeqLen", 20)
    NumHeads = hax.Axis("NumHeads", 1)
    Hid = hax.Axis("Hid", 8)

    bias = alibi_attention_bias(KeySeqLen, NumHeads)

    query = hax.ones((NumHeads, Hid))
    key = hax.ones((KeySeqLen, NumHeads, Hid))

    weights_bias = dot_product_attention_weights(Hid, KeySeqLen, query, key, bias=bias)
    weights_no_bias = dot_product_attention_weights(Hid, KeySeqLen, query, key)

    assert weights_bias.take(KeySeqLen, -1).item() > weights_bias.take(KeySeqLen, -2).item()
    assert weights_bias.take(KeySeqLen, -1).item() > weights_no_bias.take(KeySeqLen, -1).item()

    assert weights_no_bias.take(KeySeqLen, -1).item() == weights_no_bias.take(KeySeqLen, -2).item()
