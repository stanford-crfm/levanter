### Tasks
##todo
- [ ] write an overview doc for levanter
- [ ] get data and loss in Haliax


# Levanter
### Key Ideas
- Jax: Autodiff
* Equinox for torch-like goodness
* Haliax: Global-view Named Axes
* pjit for sharding
* Haliax + pjit for easier sharding
* jmp for mixed precision
* GlobalDeviceArray for sharded data loading


## Building Blocks

### Jax: Autodiff

I'm not going to go into great depth on Jax basics, because you can check out the [official Jax tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html) . But to summarize, Jax is basically a stateless, jit-optimizing version of numpy with automatic differentiation and GPU/TPU support built in. It's more than that, but those are the key features in my opinion. I will do a quick refresher on a few concepts.

#### PRNG
Randomness in Jax is carefully controlled: the "state" of a random number generator (called a `PRNGKey` in Jax) has to be passed into every invocation of an RNG or a function that calls an RNG. This adds a lot of ceremony but ensures that your code is always reproducible *and* that it
can be JIT-compiled. That looks like this:

```python
import jax.random as jrandom

key = jrandom.PRNGKey(0)
k1, k2 = jrandom.split(key, 2)

jrandom.normal(k1, (3, 4)) # 3x4 array of random numbers
jrandom.normal(k2, (3, 4)) # 3x4 array of different random numbers
```

#### vmap: Automatically adding batch axes
`vmap` is the "auto-batching" operator for Jax. It automatically vectorizes a computation so that it is applied to all sub-arrays for some new leading axis:

```python
import jax
import jax.numpy as jnp

def foo(a, b): return a @ b

a = jnp.ones((2, 3))
b = jnp.ones((3, 4))
foo(a, b) # 2x3 @ 3x4 = 2x4

batched_foo = jax.vmap(foo)
a = jnp.ones((8, 2, 3))
b = jnp.ones((8, 3, 4))
batched_foo(a, b) # 8x2x3 @ 8x3x4 = 8x2x4
```

#### scan: JIT-friendly for-loop

`scan` is a Jax primitive that allows you to write a for-loop in a functional style. For example, let's say you want
to compute the "cumulative sum" of an array. You could do this with a for-loop:

```python
def cumsum(arr):
    out = []
    for i in range(len(arr)):
        out.append(sum(arr[:i]))
    return out
```

But this is not JIT-compileable. Instead, you can use `scan`:
```python
import jax
def cumsum(arr):
    def body(carry, x):
        return carry + x, carry + x
    return jax.lax.scan(body, 0, arr)[1]
```

Scan returns both the final result of the scan and the intermediate results. In this case, we only care about the
intermediates results, so we index into the tuple with `[1]`. In Haliax we have `haliax.reduce` which is a wrapper
around `scan` that makes it easier to use and works with the NamedAxes system. Levanter also has a `reduce` function
that doesn't know about names, if you want to use it with plain Jax.

#### PyTrees

Structured computation in Jax is achieved via `PyTree`s, which is mostly just a fancy way of saying that many Jax methods
recursively apply themselves to nested data structures. By default, this includes lists, tuples, and dicts. When using
Equinox, Modules are also available as PyTrees and work in a mostly expected way. Jax has methods for flattening and
unflattening PyTrees, as well as `tree_map`, which applies a function to every leaf in a PyTree. For example:

```python
import jax
import jax.numpy as jnp

def foo(a, b): return a @ b

a = jnp.ones((2, 3))
b = jnp.ones((3, 4))
foo(a, b) # 2x3 @ 3x4 = 2x4

jax.tree_util.tree_map(foo, [a] * 5, [b] * 5) # [2x3 @ 3x4] * 5
```

Many methods in Jax are PyTree-aware, though the numpy-like API is usually not. Many methods (though for some
reason not `tree_map`) can operate on "PyTree prefixes", where the first argument is a PyTree and the rest are
prefixes of that PyTree, meaning they have the same structure up to some depth. This is used with `vmap`:

```python
import jax
import jax.numpy as jnp

def foo(args):
    a, b = args
    return a @ b

foo_vmap = jax.vmap(foo, in_axes=((0, 0), )) # vmap over the first axis for both elements of the tuple
foo_vmap2 = jax.vmap(foo, in_axes=0) # the same

a = jnp.ones((8, 2, 3))
b = jnp.ones((8, 3, 4))
foo_vmap((a, b)) # 8x2x3 @ 8x3x4 = 8x2x4
foo_vmap2((a, b)) # 8x2x3 @ 8x3x4 = 8x2x4

# don't map the second part of the tuple
foo_vmap_0 = jax.vmap(foo, in_axes=((0, None),))
```


#### pjit: distributed computation

`pjit` is the current preferred mechanism for "model parallel" and "tensor parallel" computation, and even data parallel
computation. That basic idea is that you have a "mesh" of devices, typically a 2-d grid, and you can "partition" your computation
across the mesh. The `pjit` operator takes a function and a partitioning specification, and returns a new function that
runs the distributed computation across the mesh. The partitioning specification (often 'PSpec') describes how axes of
the different arrays are partitioned across the mesh.

There's an [official pjit tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html), but I honestly don't find it that helpful... We'll do a more detailed look later on.

### Equinox: Low-magic Neural Net Library

[Equinox](https://github.com/patrick-kidger/equinox) is a neural network library for Jax that is somehow simultaneously
both less "magical" than many other neural net libraries for Jax *and* the most PyTorch-like of the bunch. It's built around
a few key ideas, but the one we're most interested in is the `Module` class. A `Module` is just a class that
has been registered with Jax as a PyTree node, which makes all of the Pytree machinery (like tree_map) work for it.
Here's a simple example:

```python
import jax
import equinox as eqx

class MyModule(eqx.Module):
    def __init__(self):
        self.param = jax.random.normal(jax.random.PRNGKey(0), (3, 4))

    def forward(self, x):
        return x @ self.param

class MyModule2(eqx.Module):
    def __init__(self):
        self.param = jax.random.normal(jax.random.PRNGKey(0), (4, 5))
        self.submodule = MyModule()

    def forward(self, x):
        return self.submodule(x) @ self.param
```

There's nothing magic about the `forward` method, or even `__init__`.

### Optax: Optimization

[Optax](https://github.com/deepmind/optax) is a library for optimization in Jax. It's basically a collection of
different gradient transformations that let you do things like gradient clipping, weight decay, and momentum. It
also has bundles of common optimization algorithms like Adam, SGD, and RMSProp. It's a pretty nice library, and
there's not that much to say.

Optimization with Equinox and Optax looks something like:

```python
import optax
import equinox as eqx

model = MyModule2()
opt = optax.adam(1e-3)
opt_state = opt.init(model)

def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

def update(model, opt_state, x, y):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
```

It's so simple that Copilot wrote all that for me... Thanks Copilot!

### Haliax: Named Tensors


Haliax is a library for named tensors in Jax. It wraps Jax's APIs (especially the numpy-like ones, along with
the core transformations like vmap, pjit, etc) to make them work with named tensors. It also builds on top of
Equinox, and adapts many of its conventions for filtering etc.

Haliax is still in development, but it's already pretty usable. Here's a simple example:

```python
import jax
from jax.random import PRNGKey
import haliax as hax

Height = hax.Axis('Height', 16)
Width = hax.Axis('Width', 16)
Batch = hax.Axis('Batch', 8)

x = hax.random.normal(PRNGKey(0), (Batch, Height, Width))

# sum out an axis
y = hax.sum(x, Height)

# sum out multiple axes
z = hax.sum(x, (Height, Width))

# broadcasting happens over named axes
normalized_x = x / hax.sum(x, Height)

# vmap over named axes. often times you may want to just skip vmap with haliax, because names are preserved etc,
# but you may still want to use it. I honestly prefer it still, but that's just me.
def foo(x):
    return hax.sum(x, Height)

foo_vmap = hax.vmap(foo, axis=Batch)
```

I took a lot of inspiration from [Alexander Rush](https://rush-nlp.com/)'s [Tensor Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor) (and [Part 2](http://nlp.seas.harvard.edu/NamedTensor)).

#### Named Axes in Jax
Jax already has some built-in support for named tensors in the form of [`xmap`](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html), which uses something like `vmap`/auto-batching to implement tensors that have both positional and named axes.XXX

## GPT-2 Implementation

You can skip this part if you're familiar with the basics of how Transformers are implemented. You might want to skim at
least the attention section to see how Haliax is used.

## Training

### Memory Use

Training a TPU

TPUs have 16GB each


### Gradient Accumulation

## jmp: Mixed Precision
## Distributed Data Loading

### ZeRO: Parameter Partitioning


## Random Gotchas
### Pjit's SPMD partitioner gets confused
Sometimes Jax doesn't shard things the way you want, and this can lead to either huge slow downs or huge memory consumption. Jax's `with_sharding_constraint` or haliax's `auto_shard` is your friend here. Sometimes a little `with_sharding_constraint` makes things worse, but more makes it better.

### Randomness isn't shardable
Unfortunately, random number generation in Jax is sequential: there's no way to "fast-forward" the RNG. Let's say I want to generate a 4096 x 8192 matrix that is sharded along the first axis across 4 nodes. Then, I might want to write something like this:

```python
import jax
from jax.experimental.pjit import pjit
from jax.random import PRNGKey
key = PRNGKey(0)

def make_matrix(key): return jax.random.normal(key, (4096, 8192))

make_matrix = pjit(make_matrix, in_axis_resources=(None,), out_axis_resources=("model", None))

my_matrix = make_matrix(key)
```

Unfortunately what Jax does here is generate all 4096 \* 8192 = 33,554,432 entries on every node, and then select the appropriate range. That is, it might look something like this:

```python
# on node 0:
whole_matrix = jax.random.normal(key, (4096, 8192))
my_shard = whole_matrix[4096/4 * 0: 4096/4 * 1, :]
# on node 1:
whole_matrix = jax.random.normal(key, (4096, 8192))
my_shard = whole_matrix[4096/4 * 1: 4096/4 * 2, :]
# etc.
```

This is pretty expensive and, more importantly, ends up using a whole lot of precisous TPU RAM, to the extent that I've had models that should fit in RAM run out of memory during initialization.

A somewhat ugly way to work around this is to use the "split" property of  something a bit less attractive like:
```python
key = PRNGKey(0)

vmap_normal = jax.vmap(lambda k: jax.random.normal(k, 8192))
my_matrix = vmap_normal(jax.random.split(key, 4096))

def make_matrix(key): return jax.random.normal(key, (4096, 8192))
```
**This changes the result from the "naive" version**, but as long as you're consistent, It's fine.

Haliax actually automatically does this under the hood along the biggest partitioned axis:

```python
import haliax as hax
Hidden = hax.Axis("Hidden", 4096)
Mlp = hax.Axis("Mlp", 8192)

key = PRNGKey(0)
my_matrix = hax.random.normal(key, (Hidden, Mlp))
```

**Note that this means that random number generation changes if you change the partitioning**, which isn't ideal, but it sometimes makes things work that didn't work before.
