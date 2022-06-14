mport os

from psithuros.named_tensors import Array, infer_named_axes, with_axis_resources

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


w1 = jnp.zeros((784, 512))
w2 = jnp.zeros((512, 10))
images = jnp.zeros((128, 784))
labels = jnp.zeros(128, dtype=jnp.int32)

# print(loss(w1, w2, images, labels))


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

loss = xmap(loss, in_axes=[...], out_axes=[...],
            axis_resources={'batch': 'x'})

devices = np.array(jax.local_devices())
with Mesh(devices, ('x',)):
  print(loss(w1, w2, images, labels))


@jit
@infer_named_axes
def named_loss2(w1: Array["inputs", "hidden"], w2: Array["hidden", "classes"], images: Array["batch", "inputs"], labels: Array["batch"]):
  predictions = named_predict(w1, w2, images)
  num_classes = lax.psum(1, 'classes')
  targets = one_hot(labels, num_classes, axis='classes')
  losses = lax.psum(targets * predictions, 'classes')
  return -lax.pmean(losses, 'batch')

loss = with_axis_resources(named_loss2, batch='x')

devices = np.array(jax.local_devices())
with Mesh(devices, ('x',)):
  print(loss(w1, w2, images, labels))