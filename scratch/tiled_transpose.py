import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jrandom
from jax import grad
from jax.experimental.host_callback import id_print
from jax.experimental.maps import xmap

key = jrandom.PRNGKey(0)

embeddings = jrandom.normal(key, shape=(4, 10, 8))

num_shards=4

def compute(embeddings, input_ids):
    hidden_states = embeddings[input_ids]
    hidden_states = lax.all_gather(hidden_states, axis_name="shard", axis=1)
    hidden_states = hidden_states.reshape(hidden_states.shape[:-2] + (-1,))

    # pretend there's a transformer here

    my_shard = lax.axis_index("shard")
    hidden_states = hidden_states.reshape(hidden_states.shape[:-1] + (num_shards, -1))
    hidden_states = hidden_states[..., my_shard, :]
    scores = hidden_states @ jnp.transpose(embeddings)
    scores = lax.psum(scores, axis_name="shard")

    return jnp.sum(jnp.sum(scores, axis=-1))

compute_xmap = xmap(compute, in_axes=[("shard", ...), (..., )], out_axes=(..., ))

compute_grad_xmap = xmap(jax.grad(compute), in_axes=[("shard", ...), (..., )], out_axes=("shard", ...))

if __name__ == "__main__":
    input_ids = jnp.array([0, 1, 2, 3, 5, 6, 7])
    print(compute_xmap(embeddings, input_ids))
    print(compute_grad_xmap(embeddings, input_ids))  # error!

