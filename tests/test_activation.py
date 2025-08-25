import numpy as np
import chex
import pytest
import jax.numpy as jnp
import haliax as hax

from levanter.utils.activation import ActivationFunctionEnum, polynorm
from test_utils import skip_if_no_torch


def _torch_polynorm_forward(x_np, weights, bias):
    """
    Defines the Torch PolyNorm class EXACTLY as provided, instantiates it,
    and returns its forward pass on x_np as a NumPy array.
    """
    import torch

    # --- DO NOT CHANGE: exact reference implementation ---
    class PolyNorm(torch.nn.Module):
        """
        A trainable activation function introduced in https://arxiv.org/html/2411.03884v1.
        The code is copied from https://github.com/BryceZhuo/PolyCom?tab=readme-ov-file/README.md
        """

        def __init__(self, eps=1e-6):
            super(PolyNorm, self).__init__()
            self.weight = torch.nn.Parameter(torch.ones(3) / 3)
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.eps = eps

        def _norm(self, x):
            return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            return (
                self.weight[0] * self._norm(x**3)
                + self.weight[1] * self._norm(x**2)
                + self.weight[2] * self._norm(x)
                + self.bias
            )

    # --- END exact reference implementation ---

    torch_poly = PolyNorm()
    with torch.no_grad():
        torch_poly.weight[:] = torch.tensor(weights, dtype=torch.float32)
        torch_poly.bias[:] = torch.tensor([bias], dtype=torch.float32)

    x_t = torch.tensor(x_np, dtype=torch.float32)
    return torch_poly(x_t).detach().cpu().numpy()


def _make_jax_array_input():
    x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = jnp.array(x_np)
    return x, x_np


def _make_named_array_input():
    A = hax.Axis("a", 2)
    B = hax.Axis("b", 3)
    x = hax.arange((A, B), dtype=jnp.float32)  # [[0,1,2],[3,4,5]]
    x_np = np.array(x.array)
    return x, x_np


@skip_if_no_torch
@pytest.mark.parametrize(
    "make_input",
    [_make_jax_array_input, _make_named_array_input],
    ids=["jax_array", "named_array"],
)
def test_polynorm_matches_reference_torch(make_input):
    x, x_np = make_input()
    weights = [0.2, 0.3, 0.5]
    bias = 0.1

    ref_np = _torch_polynorm_forward(x_np, weights, bias)
    jax_out = polynorm(x, weights, bias)

    if isinstance(x, hax.NamedArray):
        expected = hax.NamedArray(ref_np, x.axes)
    else:
        expected = ref_np

    chex.assert_trees_all_close(jax_out, expected)


def test_polynorm_enum():
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    fn = ActivationFunctionEnum.polynorm.to_fn()
    chex.assert_trees_all_close(fn(x), polynorm(x))
