# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp

from levanter.tracker import defer_tracker_for_jit, jit_log
from levanter.utils.smuggle import Smuggler


def test_smuggler_merges_nested_contexts():
    smuggler: Smuggler[dict[str, jnp.ndarray]] = Smuggler(dict)

    with smuggler.activate() as outer:
        outer["outer"] = jnp.array(1.0)
        with smuggler.activate() as inner:
            inner["inner"] = jnp.array(2.0)

        assert "inner" in outer
        assert jnp.allclose(outer["inner"], 2.0)


def test_smuggle_jit_collects_metrics():
    smuggler: Smuggler[dict[str, jnp.ndarray]] = Smuggler(dict)
    smuggled_jit = smuggler.smugglify(jax.jit)

    def fn(x):
        if smuggler.is_active:
            smuggler.get()["value"] = x
        return 2 * x

    wrapped = smuggled_jit(fn)
    metrics, result = wrapped(jnp.array(3.0))

    assert jnp.allclose(result, 6.0)
    assert jnp.allclose(metrics["value"], 3.0)


def test_smuggle_grad_collects_metrics():
    smuggler: Smuggler[dict[str, jnp.ndarray]] = Smuggler(dict)
    smuggled_grad = smuggler.smugglify(jax.grad, has_aux=True, postprocess=lambda out: (out[1], out[0]))

    def loss(x):
        value = x**2
        if smuggler.is_active:
            smuggler.get()["loss"] = value
        return value

    grad_fn = smuggled_grad(loss)
    metrics, grad = grad_fn(jnp.array(4.0))

    assert jnp.allclose(grad, 8.0)
    assert jnp.allclose(metrics["loss"], 16.0)


def test_smuggle_value_and_grad_collects_metrics():
    smuggler: Smuggler[dict[str, jnp.ndarray]] = Smuggler(dict)
    smuggled_vg = smuggler.smugglify(
        jax.value_and_grad,
        has_aux=True,
        postprocess=lambda out: (out[0][1], (out[0][0], out[1])),
    )

    def loss(x):
        value = x * (x + 1.0)
        if smuggler.is_active:
            smuggler.get()["loss"] = value
        return value

    wrapped = smuggled_vg(loss)
    metrics, (value, grad) = wrapped(jnp.array(2.0))

    assert jnp.allclose(value, 6.0)
    assert jnp.allclose(grad, 5.0)
    assert jnp.allclose(metrics["loss"], 6.0)


def test_defer_tracker_for_jit_collects_metrics():
    @jax.jit
    def do_log(x):
        with defer_tracker_for_jit() as captured:
            jit_log({"foo": x})
            return captured, x

    metrics, value = do_log(jnp.array(5.0))

    assert jnp.allclose(value, 5.0)
    assert jnp.allclose(metrics["foo"], 5.0)
