import os

from jax.interpreters.pxla import PartitionSpec

from psithuros import jax_utils

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import jax.numpy as jnp
from jax import lax, jit, pmap
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp

import jax
import numpy as np
from jax.experimental.maps import Mesh

def predict(w1, w2, images):
  hiddens = relu(jnp.dot(images, w1))
  logits = jnp.dot(hiddens, w2)
  return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(w1, w2, images, labels):
  predictions = predict(w1, w2, images)
  targets = one_hot(labels, predictions.shape[-1])
  losses = jnp.sum(targets * predictions, axis=1)
  return -jnp.mean(losses, axis=0)

dims = {
    'inputs': 784,
    'hidden': 512,
    'classes': 10,
    'batch': 128,
}

w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784))
labels = jnp.zeros(128, dtype=jnp.int32)

print(loss(w1, w2, images, labels))

from jax.experimental.pjit import pjit

import jax.random as jrandom

def init(key, sz):
  return jrandom.normal(key, shape=sz)

init_pjit=pjit(init, in_axis_resources=None, out_axis_resources=PartitionSpec('x', None), static_argnums=1)

print("q")
devices = np.array(jax.local_devices())
with Mesh(devices, ('x',)):
    print("q")
    key = jax.random.PRNGKey(0)
    print("q2")
    w1 = init_pjit(key, (dims['inputs'], dims['hidden']))
    print("q3")
    w2 = init_pjit(jax.random.PRNGKey(2), (dims['hidden'], dims['classes']))
    print("q4")
    images = init_pjit(key, (dims['batch'], dims['inputs']))
    print("q5")
    labels = jnp.zeros(dims['batch'], dtype=jnp.int32)
    print("z")

    # r = predict(w1, w2, images)
    # r
    print("qweqe")

