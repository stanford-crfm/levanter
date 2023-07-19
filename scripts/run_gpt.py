# import functools
import time

import jax
import jax.numpy as jnp
import numpy as onp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


batch_size = 256
seq_len = 2048
embed_size = 1024
vocab_size = 20000
num_layers = 20

pdrop = 0.1
USE_VMAP = True
USE_UNSAFE_RBG = True

mesh = Mesh(onp.array(jax.devices()), ("dp",))


if USE_UNSAFE_RBG:
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")

with mesh:
    key = jax.random.PRNGKey(0)

    def model(tokens, key):
        embed = jnp.take(jnp.ones((vocab_size, embed_size)), tokens, axis=0)
        # dumb fake gpt2 attn
        for i in range(0, num_layers):
            attn = jnp.einsum("...ld,...kd->...lk", embed, embed)

            if pdrop > 0.0:
                key, subkey = jax.random.split(key)
                dout = jax.random.bernoulli(subkey, pdrop, shape=attn.shape)
                attn = jnp.where(dout, jnp.zeros_like(attn), attn)

            attn = jax.nn.softmax(attn, axis=-1)
            embed = jnp.einsum("...ld,...lk->...kd", attn, embed)

        out = jnp.einsum("...ld,...kd->...lk", embed, jnp.ones((vocab_size, embed_size)))

        return out

    def compute_loss(example, key):
        pred_y = model(example, key=key)
        return jnp.mean(pred_y)

    def compute_loss_vmap(examples, key):
        key = jax.random.split(key, batch_size)
        per_ex_loss = jax.vmap(compute_loss)(examples, key)
        return jnp.mean(per_ex_loss)

    if USE_VMAP:
        compute_loss_pjit = jax.jit(compute_loss_vmap)
    else:
        compute_loss_pjit = jax.jit(compute_loss)

    # i still honestly find the way to turn a "replicated" array like batch into a sharded array to be a bit confusing
    batch = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    batch = jax.make_array_from_callback(
        (batch_size, seq_len), NamedSharding(mesh, PartitionSpec("dp", None)), lambda idx: batch[idx]
    )

    total_loss = 0.0
    total_time = 0.0

    for n in range(100):
        this_key, key = jax.random.split(key)
        time_in = time.time()
        loss = compute_loss_pjit(batch, this_key)

        total_loss += loss.item()
        time_out = time.time()

        if n > 0:
            total_time += time_out - time_in

    print(f"eval loss: {total_loss / n:.3f}")
    print(f"eval time: {total_time / (n-1):.3f}")
