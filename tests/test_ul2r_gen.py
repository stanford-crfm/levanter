import jax
from transformers import AutoTokenizer

from levanter.data.ul2r import DenoisingTaskConfig, Ul2InstanceGenerator


def test_ul2_generator():
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
