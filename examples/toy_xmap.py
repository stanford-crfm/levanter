import os

from psithuros import jax_utils

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import jax.numpy as jnp
from jax import lax, jit
from jax.nn import one_hot, relu
from jax.scipy.special import logsumexp

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
}

w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784))
labels = jnp.zeros(128, dtype=jnp.int32)

print(loss(w1, w2, images, labels))


def named_predict(w1, w2, image):
  hidden = relu(lax.pdot(image, w1, 'inputs'))
  logits = lax.pdot(hidden, w2, 'hidden')
  return logits - logsumexp(logits, 'classes')

@jit
def named_loss(w1, w2, images, labels):
  predictions = named_predict(w1, w2, images)
  num_classes = lax.psum(1, 'classes')
  targets = one_hot(labels, num_classes, axis='classes')
  losses = lax.psum(targets * predictions, 'classes')
  return -lax.pmean(losses, 'batch')


from jax.experimental.maps import xmap

in_axes = [['inputs', 'hidden', ...],
           ['hidden', 'classes', ...],
           ['batch', 'inputs', ...],
           ['batch', ...]]

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...])
print(loss(w1, w2, images, labels))

import jax
import numpy as np
from jax.experimental.maps import Mesh

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...],
            axis_resources={'batch': 'x'})

devices = np.array(jax.local_devices())
with Mesh(devices, ('x',)):
    print("distributed loss")
    print(loss(w1, w2, images, labels))

loss = xmap(named_loss, in_axes=in_axes, out_axes=[...],
            axis_resources={'batch': 'x', 'inputs': 'y'})

xy_devices = np.array(jax.local_devices()).reshape((2, -1))
with Mesh(xy_devices, ('x', 'y')):
    print("distributed loss 2")
    print(loss(w1, w2, images, labels))

import jax.random as jrandom

def init(key):
  return jrandom.normal(key, (dims['inputs'], dims['hidden']))

init = xmap(init, in_axes=[None], out_axes=(('inputs', 'hidden')), #, ('hidden', 'classes')),
            axis_resources={'inputs': 'x', 'hidden': 'y'}, axis_sizes=dims)


with Mesh(xy_devices, ('x', 'y')):
  key = jax.random.PRNGKey(0)
  # result = init(jax_utils.shaped_rng_split(key, (dims['inputs'], dims['hidden'])))
  result = init(key)


if __name__ == '__main__':
    pass