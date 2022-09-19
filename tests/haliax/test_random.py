import jax

import haliax as hax


def test_empty_shape():
    key = jax.random.PRNGKey(0)
    hax.random.uniform(key, shape=())
