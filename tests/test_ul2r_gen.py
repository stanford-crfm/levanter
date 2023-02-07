import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

import haliax as hax
from levanter.data.ul2r import DenoisingTaskConfig, Ul2Example, Ul2InstanceGenerator


def test_ul2_generator_seed_works():
    # Generate synthetic data
    synthetic_data = jax.random.randint(jax.random.PRNGKey(0), shape=(50, 512), minval=0, maxval=1000)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ul2_generator = Ul2InstanceGenerator(
        tokenizer,
        [f"<mask_{i}>" for i in range(500)],
        DenoisingTaskConfig.ul2r_configs(),
    )

    for i, tokens in enumerate(synthetic_data):
        a = ul2_generator.sample(tokens, jax.random.PRNGKey(i)).render(tokenizer)
        b = ul2_generator.sample(tokens, jax.random.PRNGKey(i)).render(tokenizer)
        assert a == b
        c = ul2_generator.sample(tokens, jax.random.PRNGKey(i + 1)).render(tokenizer)
        assert a != c


def test_ul2_generator_can_handle_too_few_sentinels():
    synthetic_data = jax.random.randint(jax.random.PRNGKey(0), shape=(10, 1000), minval=0, maxval=1000)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ul2_generator = Ul2InstanceGenerator(
        tokenizer,
        [f"<mask_{i}>" for i in range(2)],
        DenoisingTaskConfig.ul2r_configs(),
    )

    for i, tokens in enumerate(synthetic_data):
        # just make sure it doesn't crash
        ul2_generator.sample(tokens, jax.random.PRNGKey(i))


def test_decoder_only_example():
    QLen = hax.Axis("QLen", 25)
    KLen = QLen.alias("KLen")

    example = Ul2Example(task_token=1000, inputs=np.arange(10), outputs=np.arange(20, 30))

    converted = example.to_decoder_only(1001, QLen, KLen)

    tokens = converted.tokens.array

    assert tokens[0] == 1000
    assert tokens[1] == 0
    assert np.all(tokens[1:11] == example.inputs)
    assert np.all(tokens[11:21] == example.outputs)
    assert np.all(tokens[21:] == 1001)

    loss_mask = converted.loss_mask.array

    assert np.sum(loss_mask) == len(example.outputs)
    assert np.all(loss_mask[0:10] == 0)
    assert np.all(loss_mask[10:20] == 1)
    assert np.all(loss_mask[20:] == 0)

    attn_mask = converted.attn_mask.rearrange((QLen, KLen)).array

    assert hax.all(hax.sum(converted.attn_mask, QLen) > 0)
    assert hax.all(hax.sum(converted.attn_mask, KLen) > 0)

    assert np.all(attn_mask[:, 0] == 1)
    assert np.all(np.sum(attn_mask[np.arange(0, 11), :], 1) == 11)  # inputs fully attend to task token + inputs
    # start with 1 extra because you can attend to yourself
    assert np.all(
        np.sum(attn_mask[np.arange(11, 21), :], 1) == 11 + np.arange(1, 11)
    )  # outputs attend to task token + inputs + previous outputs


# to make double extra sure, verify we don't leak information on accident
def test_decoder_only_example_attention():
    L = 20
    D = 2
    SeqLen = hax.Axis("SeqLen", L)
    KSeqLen = SeqLen.alias("KSeqLen")
    Head = hax.Axis("Head", D)

    input_length = 10

    inputs = np.arange(input_length)
    outputs = np.arange(input_length * 2, (input_length * 2) + (L - input_length))
    assert len(outputs) + input_length == L

    ex = Ul2Example(task_token=1000, inputs=inputs, outputs=outputs).to_decoder_only(1001, SeqLen, KSeqLen)

    attn_mask = ex.attn_mask

    # testing here that we can't attend to the inputs from the outputs
    keys = np.zeros((L, D), dtype=np.float32)
    keys[input_length + 1, 1] = 100.0  # really want to attend to this
    values = np.zeros((L, D), dtype=np.float32)
    values[input_length + 1, 1] = 300.0  # check if we did attend

    query = np.ones((L, D), dtype=np.float32)

    query = hax.named(query, (SeqLen, Head))
    keys = hax.named(keys, (KSeqLen, Head))
    values = hax.named(values, (KSeqLen, Head))

    result = hax.nn.attention.dot_product_attention(SeqLen, KSeqLen, Head, query, keys, values, mask=attn_mask)
    result = result.rearrange((SeqLen, Head)).array
    # the values for the outputs should all close to 300
    assert jnp.allclose(result[input_length + 1 :, 1], 300)
    assert jnp.allclose(result[0 : input_length + 1, 1], 0)
