from functools import partial

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from jax import pmap
from optax import OptState
from transformers import GPT2Config

from psithuros.gpt2 import Gpt2LMHeadModel

NUM_TOKENS = 2048
SEQ_LEN = 512

# cf https://github.com/google-research/language/blob/aa58066bec83d30de6c8f9123f0af7b81db3aeba/language/mentionmemory/training/trainer.py

def replicate(tree, devices=None):
    """Replicates arrays to multiple devices.
    Args:
      tree: a pytree containing the arrays that should be replicated.
      devices: the devices the data is replicated to
        (default: same order as expected by `jax.pmap()`).
    Returns:
      A new pytree containing the replicated arrays.
    """
    return jax.device_put_replicated(tree, devices or jax.devices())


def dataloader(arrays, batch_size, *, key, max_passes=None):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    i = 0
    while max_passes is None or i < max_passes:
        i += 1
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, *, key):
    k_x, k_y = jrandom.split(key, 2)
    x = jrandom.randint(k_x, [dataset_size, SEQ_LEN], minval=0, maxval=NUM_TOKENS)
    # y = jrandom.randint(k_y, [dataset_size, SEQ_LEN], minval=0, maxval=NUM_TOKENS)
    y = jnp.concatenate( [x[:, 1:], jnp.zeros((dataset_size, 1), dtype=jnp.int32)], axis=1)

    return x, y


def main(
    dataset_size=10000,
    batch_size=256,
    learning_rate=3e-3,
    steps=200,
    seed=5678,
):
    data_key, loader_key, model_key, training_key = jrandom.split(jrandom.PRNGKey(seed), 4)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    config = GPT2Config(vocab_size=NUM_TOKENS, n_positions=SEQ_LEN, n_embd=128, n_ctx=SEQ_LEN, n_layer=4, n_head=4, n_embd_shared_axes=0, hidden_dim=128, num_attention_heads=4, intermediate_size=128, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=SEQ_LEN, type_vocab_size=2, initializer_range=0.02)

    model = Gpt2LMHeadModel(config, key=model_key)

    def compute_loss(model, x, y, inference, key):
        model = partial(model, inference=inference, key=key)
        pred_y = jax.vmap(model)(x)
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(y, num_classes=NUM_TOKENS)))

    # compute_loss_and_grad = pmap(compute_loss, "device", in_axes=(None, 0, 0, None, None), static_broadcasted_argnums=(3,))
    compute_loss_and_grad = eqx.filter_value_and_grad(compute_loss)

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region
    def make_step(model, x, y, opt_state, key):
        loss, grads = compute_loss_and_grad(model, x, y, False, key)
        loss = lax.pmean(loss, "device")
        grads = lax.pmean(grads, "device")
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    make_step = pmap(make_step, "device", in_axes=(0, 0, 0, 0, 0))



    devices = jax.devices()
    assert batch_size % len(devices) == 0
    # TODO: need to replicate the key?

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)

    model = replicate(model)
    opt_state = replicate(opt_state)

    keys = jax.random.split(training_key, steps)

    for step, (x, y), k in zip(range(steps), iter_data, keys):
        k = jax.random.split(k, len(devices))
        x = x.reshape([len(devices), batch_size//len(devices), SEQ_LEN])
        y = y.reshape([len(devices), batch_size//len(devices), SEQ_LEN])
        loss, model, opt_state = make_step(model, x, y, opt_state, k)
        loss = jnp.mean(loss).item()
        print(f"step={step}, loss={loss}")

    # test:
    total_loss = 0.0

    # @eqx.filter_jit
    # def compute_loss_test(model, x, y):
    #     model = partial(model, inference=True, key=None)
    #     pred_y = jax.vmap(model)(x)
    #     return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(y, num_classes=NUM_TOKENS)))

    test_loader = dataloader((xs, ys), batch_size, max_passes=1, key=loader_key)

    compute_loss = pmap(compute_loss, "device", in_axes=(0, 0, 0, 0, 0), static_broadcasted_argnums=(3))
    for (x, y) in test_loader:
        key = jax.random.split(training_key, len(devices))
        x = x.reshape([len(devices), batch_size//len(devices), SEQ_LEN])
        y = y.reshape([len(devices), batch_size//len(devices), SEQ_LEN])
        loss = compute_loss(model, x, y, True, key)
        total_loss += jnp.sum(loss).item()

    # num_correct = jnp.sum((pred_ys > 0.5) == ys)
    # final_accuracy = (num_correct / dataset_size).item()
    # print(f"final_accuracy={final_accuracy}")


if __name__ == "__main__":
    main()
