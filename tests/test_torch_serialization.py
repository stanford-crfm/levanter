import equinox as eqx
import jax
import jax.experimental
import jax.experimental.mesh_utils
import numpy as np
import pytest

import haliax as hax

from levanter.compat.torch_serialization import (
    StateDictSerializationMixin,
    flatten_linear_layers,
    jax_tree_from_state_dict,
    to_numpy_state_dict,
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


class SerializableModel(StateDictSerializationMixin, eqx.Module):
    weight: hax.NamedArray

    def __init__(self, weight: hax.NamedArray):
        self.weight = weight
        super().__init__()

    def __call__(self, x):
        return hax.dot(self.weight.axes[0], self.weight, x)


def test_to_numpy_state_dict():
    # from jax.experimental.topologies import get_topology_desc
    # devices = get_topology_desc(
    #     platform="tpu",
    #     topology_name="v5e:8x8",
    #     chip_config_name="default",
    #     chips_per_host_bounds=(2, 2, 1),
    #     num_slices=1,
    # ).devices
    # print(devices)

    device_layout = jax.experimental.mesh_utils.create_device_mesh([4, 8])
    device_mesh = jax.sharding.Mesh(device_layout, ("x", "y"))

    a = hax.Axis("A", 2)
    b = hax.Axis("B", 3)
    c = hax.Axis("C", 4)
    w = hax.arange(hax.Axis("D", 2 * 3 * 4))
    w = hax.unflatten_axis(w, "D", (a, b, c))
    w = hax.shard_with_axis_mapping(w, {"A": "x", "B": "y"}, device_mesh)
    print("Created weight:", w, w.shape)
    model = SerializableModel(w)

    def _test_copy_to_cpu(chunk_size):
        state_dict = to_numpy_state_dict(model, copy_chunk_size=chunk_size)
        assert isinstance(state_dict, dict)
        weights = state_dict["weight"]
        assert weights.shape == (2, 3, 4)
        assert np.all(weights == np.arange(2 * 3 * 4).reshape(2, 3, 4))

    _test_copy_to_cpu(1)
    _test_copy_to_cpu(2)
    _test_copy_to_cpu(3)
    _test_copy_to_cpu(7)
    _test_copy_to_cpu(100)
