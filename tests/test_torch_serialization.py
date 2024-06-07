import jax
import pytest

import haliax as hax

from levanter.compat.torch_serialization import (
    flatten_linear_layers,
    jax_tree_from_state_dict,
    unflatten_linear_layers,
)


@pytest.mark.parametrize("out_dims_first", [True, False])
def test_unflatten_linear_layers(out_dims_first: bool):
    H = hax.Axis("H", 10)
    W = hax.Axis("W", 20)
    D = hax.Axis("D", 30)
    B = hax.Axis("B", 40)
    linear = hax.nn.Linear.init((H, W), (D, B), key=jax.random.PRNGKey(0), use_bias=True, out_first=False)

    assert linear.weight.axes == (H, W, D, B)

    # first flatten the weight matrix
    flat = flatten_linear_layers(None, linear, out_dims_first_in_dict=out_dims_first)
    if out_dims_first:
        assert flat["weight"].shape == (D.size * B.size, H.size * W.size)
    else:
        assert flat["weight"].shape == (H.size * W.size, D.size * B.size)
    assert flat["bias"].shape == (D.size * B.size,)
    assert flat["weight"].dtype == flat["bias"].dtype == linear.weight.dtype

    # now unflatten it
    unflat_dict = unflatten_linear_layers(None, flat, linear, out_dims_first_in_dict=out_dims_first)
    new_linear = jax_tree_from_state_dict(linear, unflat_dict)

    assert new_linear.weight.axes == (H, W, D, B)
    assert new_linear.bias.axes == (D, B)
