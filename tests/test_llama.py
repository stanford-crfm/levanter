from jax import random

from levanter.models.llama import LlamaRotaryEmbedding


def test_llama_rotary_embedding():
    dim = 2048
    max_position_embeddings = 2048
    seq_len = 2048
    base = 10000
    key = random.PRNGKey(0)
    rotary_emb = LlamaRotaryEmbedding(dim=dim)
    rotary_emb.setup()
    x = random.normal(key, (1, 2048))
    x = rotary_emb(x, seq_len=seq_len)

