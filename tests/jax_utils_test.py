import jax
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import psithuros.jax_utils as jax_utils
import psithuros.models.gpt2
from psithuros.axis_names import Array


def test_backward_shape_jit_has_same_shape():
    def g(x):
        return jnp.sin(jnp.sin(x))

    @jax.jit
    def f(x):
        return g(x)

    graph_size_g = jax_utils.backward_graph_size(g, jnp.ones(1,))

    assert graph_size_g == 2

    graph_size_f = jax_utils.backward_graph_size(f, jnp.ones(1,))

    assert graph_size_f == 2


def test_backward_shape_remat_sin2_holds_no_internals():
    def g(x):
        return jnp.sin(jnp.sin(x))

    @jax.remat
    def f(x):
        return g(x)

    graph_size_g = jax_utils.backward_graph_size(g, jnp.ones(1,))
    graph_size_f = jax_utils.backward_graph_size(f, jnp.ones(1,))

    assert(graph_size_g == 2)
    assert(graph_size_f == 0)


def test_backward_shape_simple_linear():
    X = jnp.ones((2, 3))
    def g(x):
        return X @ x

    @jax.jit
    def f(x):
        return g(x)

    graph_size_g = jax_utils.backward_graph_size(g, jnp.ones((3, 1)))

    assert graph_size_g == 0

    # TODO: unfortunately it's tricky to track the "X" when we jit.
    # graph_size_f = jax_utils.backward_graph_size(f, jnp.ones((3, 1)))
    #
    # assert graph_size_f == 0


def test_backward_shape_linear():
    linear = eqx.nn.Linear(5, 5, key=jrandom.PRNGKey(0))

    graph_size = jax_utils.backward_graph_size(linear, jnp.ones(5))
    assert graph_size == 0


def test_backward_shape_sigmoid():
    graph_size_mlp = jax_utils.backward_graph_size(jax.nn.sigmoid, jnp.ones((1,)))

    # ATM, jax stores x and 1 - x in the graph, which is a little silly
    assert graph_size_mlp == 2


def test_backward_shape_mlp_relu():
    mlp = psithuros.models.gpt2.Gpt2Mlp(5, 3, activation_fn="relu", key=jrandom.PRNGKey(0))

    # graph_size_mlp = jax_utils.backward_graph_size(mlp, jnp.ones((5,)))
    graph_size_mlp = jax_utils.backward_graph_size(mlp, jnp.ones((5,)))

    # 3 for output of relu + an unnecessary 3 for the zeros array from relu
    assert(graph_size_mlp == 2 * 3)

def test_backward_shape_mlp_gelu_approx():
    mlp = psithuros.models.gpt2.Gpt2Mlp(1600, 6400, activation_fn="gelu_new", key=jrandom.PRNGKey(0))

    # graph_size_mlp = jax_utils.backward_graph_size(mlp, jnp.ones((5,)))
    graph_size_mlp = jax_utils.backward_graph_size(mlp, jnp.ones((1600,)))

    jax_utils.dump_fwd_bwd_jaxprs("mlp_gelu_new", lambda mlp, x: jnp.sum(mlp(x)), mlp, jnp.ones((1600,)))

    flop_estimate = jax_utils.flops_estimate(jax.value_and_grad(lambda x: jnp.sum(mlp(x))), jnp.ones((1600,)))
    print(flop_estimate)

    # 3 for output of relu + an unnecessary 3 for the zeros array from relu
    assert graph_size_mlp == 2 * 3

def test_backward_shape_softmax():
    sz = jax_utils.backward_graph_size(jax.nn.softmax, jnp.arange(5, dtype=jnp.float32))

    assert sz == 12



if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
