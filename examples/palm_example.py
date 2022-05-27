import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from optax import OptState

from psithuros.models.palm_lite import PaLM

NUM_TOKENS = 2048
SEQ_LEN = 512

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
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
    x = jrandom.randint(key, [dataset_size, SEQ_LEN], minval=0, maxval=NUM_TOKENS)
    y = x + 1

    return x, y


def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    seed=5678,
):
    data_key, loader_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = PaLM(depth=4,
                 num_tokens=NUM_TOKENS,
                 dim=128,
                 dim_head=128,
                 heads=4,
                 key=model_key,
                 ff_mult=4,
                 max_seq_len=SEQ_LEN,
                 )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x)
        # Trains with respect to binary cross-entropy
        return jnp.mean(optax.softmax_cross_entropy(pred_y, jax.nn.one_hot(y, num_classes=NUM_TOKENS)))
        # return jnp.mean(

    # Important for efficiency whenever you use JAX: wrap everything into a single JIT
    # region.
    @eqx.filter_jit
    def make_step(model, x, y, opt_state: OptState):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


if __name__ == "__main__":
    main()
