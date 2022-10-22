class PartitionSpec:
pass### Tasks
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
from jax.random import PRNGKey
import haliax as hax  # this alias amuses me

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

By convention, we capitalize the names of axes. This is because it makes it easier to visually distinguish them.


#### Why Named Arrays?

I get really confused when reading tensor code that uses axes like `0`, `1`, `2`, etc. It's not clear what
those axes are, and it's especially unclear when you have multiple tensors with different shapes. I've also been
bitten by implicit broadcasting way too many times.

So I found [Alexander Rush](https://rush-nlp.com/)'s [Tensor Considered Harmful](http://nlp.seas.harvard.edu/NamedTensor)
(and [Part 2](http://nlp.seas.harvard.edu/NamedTensor)) to be very convincing. Named arrays are a way to make this code
more readable, and more robust.

Named Arrays will also make it easier to write partitioned models: jax's `pjit` operator works over "meshes" of devices,
and you partition your parameters and computation along array axes across the mesh. Named arrays make it easier to map
semantically meaningful axes to the mesh axes. (See below for more on this.)

#### Named Axes in Jax
Jax already has some built-in support for named tensors in the form of [`xmap`](https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html), which uses something like `vmap`/auto-batching to implement tensors that have both positional and named axes.
I was super excited about `xmap` when I first heard about it, but 1) they seem to be deprioritizing it in favor of `pjit`
and 2) ultimately `xmap` can be confusing because you write non-named code for positional axes, then add names "outside"
of the main model code itself. I think it's ultimately harder to reason about than named tensors that are fully integrated,
and it makes it harder to play with different partitioning strategies.


## GPT-2 Implementation

You can skip this part if you're familiar with the basics of how Transformers are implemented. You might want to skim at
least the attention section to see how Haliax is used.

The whole implementation is [here](https://www.github.com/stanford-crfm/levanter/blob/main/levanter/models/gpt2.py),
(If you look at the whole thing, I caution you to skip over the torch serialization compatibility parts because they're
messy and not that interesting for our purposes here.) I'll walk through the more interesting bits. In terms of basic
structure and grouping of classes it's pretty similar to standard PyTorch implementations.

XXX this comes across as even more defensive than I intended
If you want, you can compare it with Andrei Karpathy's [mingpt implementation](https://github.com/karpathy/minGPT/blob/master/mingpt/model.py)
. Ours is a bit longer (even excluding the torch serialization parts) for a few reasons:
* Boilerplate from declaring fields for modules
* More type annotations
* Tricks to improve stability from the [Mistral project](https://crfm.stanford.edu/2021/08/26/mistral.html#eureka/)


### Attention
Let's jump right into the attention module, starting with the preamble

```python
import haliax as hax
from haliax import Axis, NamedArray
import equinox as eqx
import haliax.nn as hnn

class Gpt2Attention(eqx.Module):
    c_attn: hnn.Linear  # input projection from [embed] -> [(q, k, v), heads, head_dim]
    c_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    dropout: hnn.Dropout

    causal: bool = eqx.static_field()
    SeqLen: Axis = eqx.static_field()
    HeadDim: Axis = eqx.static_field()
    Heads: Axis = eqx.static_field()
    Qkv: Axis = eqx.static_field()

    # Mistral stability tweaks
    scale_by_inverse_layer_idx: bool = eqx.static_field()
    upcast: bool = eqx.static_field()
```

The `static_field` decorator is a way to declare fields that are static, and don't change during training. (They're also
not parameters, so they don't get updated during training.) `causal` means that this is a causal attention layer, which
means that the queries can only attend to the past.

Here's the `__init__` method. It's just initializing the parameters and setting the static fields, but it illustrates
some Jax/Haliax idioms.

```python
    def __init__(
        self,
        SeqLen: Axis,
        InDim: Axis,
        Heads: Axis,
        HeadDim: Axis,
        dropout_prob: float,
        scale_by_inverse_layer_idx: bool,
        upcast: bool,
        *,
        key,
        causal: bool = True,
    ):
        self.Heads = Heads
        self.HeadDim = HeadDim
        self.SeqLen = SeqLen
        self.Qkv = Axis("qkv", 3)
        self.KeySeqLen = SeqLen.alias("key_" + SeqLen.name)

        k_c, k_proj = jrandom.split(key, 2)  # splitting random keys is how you get different random numbers from different calls
        # Haliax's Linear allows you to specify multiple input and output axes, and it will do the right thing
        # I find this clearer than the reshape heavy code you usually see
        self.c_attn = hnn.Linear(In=InDim, Out=(self.Qkv, self.Heads, self.HeadDim), key=k_c)
        self.c_proj = hnn.Linear(In=(self.Heads, self.HeadDim), Out=InDim, key=k_proj)
        self.dropout = hnn.Dropout(dropout_prob)

        self.causal = causal
        self.scale_by_inverse_layer_idx = scale_by_inverse_layer_idx
        self.upcast = upcast
```

The basic flow of multi-headed attention in a standard transformer is:
1. For each position, project the input embedding into `Heads` heads, each of which with a query, key, and value vector of dim `HeadDim`. This yields three tensors of shape `[SeqLen, Heads, HeadDim]`.
2. Compute the attention scores for each `(head, position_in_query, position_in_key)`.  This yields a tensor of shape `[SeqLen, Heads, SeqLen]`.
3. Normalize the attention scores to get the attention weights. This yields a tensor of shape `[SeqLen, Heads, SeqLen]`.
4. Compute the attention output for each `(head, position)` by summing up the value vectors, weighting them by the attention weights. This yields a tensor of shape `[SeqLen, Heads, HeadDim]`.
5. Project the attention output back to the embedding dimension. This yields a tensor of shape `[SeqLen, Embed]`.

Let's see how this looks in Haliax:

```python
def __call__(self, hidden_states: NamedArray, layer_idx, inference: bool = True, *, key):
    # 1. Project the input to [seqlen, heads, head_dim]
    qkv_out = self.c_attn(hidden_states)
    q, k, v = qkv_out.unbind(self.Qkv)

    # Rename k and v's SeqLen as haliax doesn't support unnamed axes or duplicate axes
    k = k.rename({self.SeqLen: self.KeySeqLen})
    v = v.rename({self.SeqLen: self.KeySeqLen})

    # mistral tweak: scale norms by 1/sqrt(layer_idx) to prevent blowup
    scale = jax.lax.rsqrt(float(self.HeadDim.size))
    if self.scale_by_inverse_layer_idx:
        scale /= layer_idx + 1.0

    # do this first to help keep FP values small. Papers usually show this after the dot product.
    q = q * scale

    # mistral tweak: attention scores can overflow FP16, or just be too imprecise, so upcast to FP32
    if self.upcast:
        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)

    # 2. compute attention scores [batch, heads, seq_len, key_seq_len]
    attn_scores = hax.dot(self.HeadDim, q, k)

    if self.causal:
        causal_mask = hax.tril(hax.ones((self.SeqLen, self.KeySeqLen), dtype=bool), self.SeqLen, self.KeySeqLen)
        attn_scores = hax.where(causal_mask, attn_scores, -1e9)

    # 3. normalize attention scores to "attention weights"
    attn_weights = hnn.softmax(attn_scores, axis=self.KeySeqLen).astype(hidden_states.dtype)
    attn_weights = self.dropout(attn_weights, key=key, inference=inference)

    # 4. compute attention output by summing up the values weighted by the attention scores
    attn_output = hax.dot(self.KeySeqLen, attn_weights, v)  # [heads, seq_len, head_dim]

    # 5. project the attention output back to the original input dimension
    attn_output = self.c_proj(attn_output)
    return attn_output
```

If you're not used to the `tril`-as-a-mask trick, it's a way to create a causal mask so that the attention scores
for a query can only attend to the past. The `tril` function creates a lower triangular matrix. It's equivalent to:
```python
causal_mask = jnp.zeros(seq_len, key_seq_len)
for i in range(seq_len):
    for j in range(key_seq_len):
        if j >= i:
            mask[i, j] = 1
```

### The MLP

The MLP is not terribly interesting. It's just a linear layer followed by a GELU activation, followed by another linear layer.

```python
class Gpt2Mlp(eqx.Module):
    act: Callable = eqx.static_field()
    c_fc: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    c_proj: hnn.Linear  # projection from Intermediate to Embed

    def __init__(self, Embed: Axis, Intermediate: Axis, activation_fn, *, key):
        k_fc, k_proj = jrandom.split(key, 2)
        self.c_fc = hnn.Linear(Out=Intermediate, In=Embed, key=k_fc)
        self.c_proj = hnn.Linear(Out=Embed, In=Intermediate, key=k_proj)
        self.act = ACT2FN[activation_fn]  # type: ignore

    def __call__(self, hidden_states: NamedArray):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = jax.tree_util.tree_map(self.act, hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states
```

### The Block

The block is the basic unit of the transformer. It's just a multi-headed attention layer, followed by a layer norm, followed by an MLP, followed by another layer norm.

```python
class Gpt2Block(TorchSerializationMixin, eqx.Module):
    ln_1: hnn.LayerNorm
    attn: Gpt2Attention
    ln_2: hnn.LayerNorm
    mlp: Gpt2Mlp
    resid_dropout: hnn.Dropout

    def __init__(self, config: Gpt2Config, *, key):
        # skipping this because it's boring
        ...

    def __call__(self, hidden_states: NamedArray, inference, layer_idx, *, key):
        k1, k2, k3 = haliax.jax_utils.maybe_rng_split(key, 3)

        attn_output = self.attn(self.ln_1(hidden_states), inference=inference, layer_idx=layer_idx, key=k1)
        attn_output = self.resid_dropout(attn_output, key=k2, inference=inference)
        hidden_states = hidden_states + attn_output

        ff_output = self.mlp(self.ln_2(hidden_states))
        ff_output = self.resid_dropout(ff_output, key=k3, inference=inference)
        hidden_states = hidden_states + ff_output

        return hidden_states
```

Probably the least understandable thing here is the `maybe_rng_split` function. It's a helper function that
splits the key if it's a JAX PRNG key, and returns a list of None if it's `None`. This is because the `key` argument
is optional, and if it's not provided, we don't want to split the key. This is useful for inference time when we
don't use dropout.

### The Transformer

The transformer conceptually is just a stack of these blocks (plus a final layer norm). In Jax, we can use `jax.vmap` to
create a vectorized block stack, and then use `jax.lax.scan` to apply the blocks in sequence. We use Haliax named
variants of these functions: `hax.vmap` and `hax.fold`. (Jax doesn't have `fold` per se, but instead uses `scan` for both.
Haliax also has a `hax.scan` function that's equivalent to `jax.lax.scan`.)

```python

This can be a bit hard to understand, so let's break it down. First, we create a vectorized block stack:
```python
class Gpt2Transformer(eqx.Module):
    config: Gpt2Config = eqx.static_field()
    blocks: Gpt2Block
    ln_f: hnn.LayerNorm

    @property
    def Layers(self) -> Axis:
        return self.config.Layers

    def __init__(self, config: Gpt2Config, *, key):
        super().__init__()
        self.config = config

        self.blocks = hax.vmap(Gpt2Block, self.Layers)(config, key=shaped_rng_split(key, config.num_layers))
        self.ln_f = hnn.LayerNorm(config.Embed, eps=config.layer_norm_epsilon)
```

Recall that `vmap` inserts a new axis into the function. So, `hax.vmap(Gpt2Block)` creates a function that takes
a "batch" of keys (meaning a key array with a leading axis for the number of layers) and returns a "batch" of blocks
(meaning a single Block object whose arrays have a leading axis for the number of layers).

Next, we create a function that applies the blocks in sequence:

```python
    def __call__(self, hidden_states: NamedArray, inference, *, key) -> NamedArray:
        def do_block(hidden_states, block, layer_idx, key):
            return block(hidden_states, inference=inference, layer_idx=layer_idx, key=key)

        keys = hax.jax_utils.maybe_rng_split(key, self.config.num_layers) if key is not None else None
        hidden_states = hax.fold(do_block, axis=self.Layers)(  # type: ignore
            hidden_states, self.blocks, hax.arange(self.Layers), key=keys  # type: ignore
        )
        hidden_states = self.ln_f(hidden_states)

        return hidden_states
```

If you're not used to functional programming, this might be a bit hard to understand. The `fold` function is
equivalent to a `for` loop. It takes a function `do_block` and applies it to each element of the `self.blocks` array,
accumulating the result. The `axis` argument tells it which axis to iterate over. This invocation is equivalent to the
following Python:

```python
hidden_states = hidden_states
for layer_idx, block in enumerate(self.blocks):
    key = keys[layer_idx] if keys is not None else None
    hidden_states = do_block(hidden_states, block, layer_idx, key)
```

The reason to use fold/scan in Jax is that it makes compilation much faster, and it also works with non-constant
lengths. We're mostly using it for compilation speed. Eventually we'll add support for fancy gradient checkpointing
strategies to Haliax `fold` and `scan`.

## First cut at training: simple data parallel training via `pmap`

XXX TODO:

## Reducing memory usage

When doing training, you need to store the parameters for your model, the gradients, and the optimizer state. For Adam
and its kin, this is another two copies of the parameters. If you store your parameters in FP32, you need at a minimum
16 * `num_params` bytes of memory, without including the memory needed for activations. (For LLMs, the data for a batch
is typically trivial in comparison.) If you don't use ZeRO (XXX cite) or some other technique, you'll need to store all
of these on every TPU/GPU. It's generally recommended to store all of these (maybe not the gradients) in FP32 for training.

While we'll get to ZeRO/FSDP in a bit, in the meantime we can use mixed precision training to reduce the memory footprint
of the model.

A single v3 or v4 TPU core has 16GB of memory. This means that, for a 345M parameter model, we've already committed
to using 5.5GB of memory for the parameters and optimizer states alone. This is plenty of space, but if we go to
a 750M parameter model, we'll need 12GB of memory. This doesn't give us a ton of room for the activations, which
can be nontrivial.

### Gradient Checkpointing



### Mixed Precision Training via `jmp`


## Model and Activation Partitioning via `pjit` (and data parallelism revisited)

Activation partitioning is a technique that allows you to split up the activations of a model across multiple devices.
Typically, this is accompanied by model partitioning, which splits up the parameters of the model across multiple devices.
(This is distinct from ZeRO, which also splits up the model parameters and associated states. We'll cover the difference
later.)

Jax has a principle that computation follows data: if you want to do a distributed matrix multiplication, you can split
your matrix up across multiple devices, do the appropriate local matrix multiplications, and then aggregate the results.
As an example, if I have a 1024x512 matrix and I want to do a distributed matrix multiply by a hidden state vector of
dimension 512, I can split the matrix up into 4 256x512 matrices, do 4 local matrix multiplies each against the vector,
and then sum up the resulting 1024 vectors.

This is the principle behind `pjit`, except it's all done automatically under the hood. `pjit` will let us write our model
code as if it were running on a single device, but then we specify how we want our parameters and data to be partitioned
across devices, and `pjit` and the [XLA SPMD partitioner](https://arxiv.org/pdf/2105.04663.pdf) will do the rest.

### Device Meshes

Currently in Jax, the primary abstraction for partitioning data across a device is a "device mesh", which is
basically just an N-dimensional array of devices. Typically, we use 2-dimensional meshes, and the two axes of the
mesh are labeled "data" and "model". The "data" axis is the one that we'll use for data parallelism, and the "model"
axis is the one we'll use for model parallelism.

Given a device mesh, the way to partition a device in Jax is to specify a `PartitionSpec`, which is a namedtuple
of mesh axis names (or `None`) that is the same length as the shape of the array. For example, if we have a 2x2
device mesh, we can partition a 4x4 array across the mesh by specifying a `PartitionSpec` of `("data", "model")`.
If you have an array with more dimensions than that, you can specify `None` for the extra dimensions. For example,
if you have a 2x2 device mesh and you want to partition a 4x4x4 array across the mesh, you can specify a
`PartitionSpec` of `("data", "model", None)`. Axes that are `None` will be replicated across the mesh.
(One can also partition an array along multiple axes by specifying
a tuple of axis names, but we won't cover that here.)

### Jax `pjit`

Now at long last, we can get to `pjit`. `pjit` is a function that takes a function and a `PartitionSpec` for each
input (or None to indicate that the input should be fully replicated across the mesh) and for each output,
and returns a function that runs the function after partitioning the data across the mesh (if they need to be).

Here's an example:
```python
from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec
import jax
import jax.numpy as jnp
import numpy as onp

def matmul(x, y):
    return x @ y

x = jnp.ones((512, 128))
y = jnp.ones((128, 1))

pjit_matmul = pjit(matmul,
                   in_axis_resources=(PartitionSpec("model", None), None),
                   out_axis_resources=PartitionSpec("model"))

# assume we have 8 devices, e.g. a v3-8 TPU node
devices = onp.array(jax.devices()).reshape((4, 2))
mesh = Mesh(devices, ("model", "data"))

with mesh:
    print(pjit_matmul(x, y))
```


This will divide the 512x128 matrix `x` into 4 128x128 matrices, with each submatrix replicated across the 2 devices
in the "data" axis. 'y' will be fully replicated across the mesh. The matrix multiply will be done locally on each
device, and the results will continue to be partitioned across the mesh. The final result will be a 512x1 vector,
which is partitioned across the "model" axis, so the 128x1 vectors will be replicated across the 2 devices in the
"data" axis.

Now consider an alternative sharding strategy, where we shard the 512x128 matrix on the second axis instead of the
first:
```python
pjit_matmul = pjit(matmul,
                   in_axis_resources=(PartitionSpec(None, "model"), PartitionSpec("model", None)),
                   out_axis_resources=PartitionSpec(None))

with mesh:
    print(pjit_matmul(x, y))
```

This will divide the 512x128 matrix `x` into 4 512x32 matrices, with each submatrix replicated across the 2 devices
in the "data" axis. `y` will also be partitioned across the "model" axis, so each device will have a 32x1 vector.
The matrix multiply will be done locally on each device, and then *summed* across the mesh. The final result will still
be a 512x1 vector, which is fully replicated across the mesh.

Just to drive the point home: you can change a computation from "fan-in" to "fan-out" by changing the partitioning.
This is a really powerful property of `pjit` and the XLA SPMD partitioner.

### Haliax `named_pjit`

`pjit` is great, but it can be a bit of a pain to use when you have a lot of inputs and outputs. For example, if you
have a complex nested model hierarchy (e.g. our GPT-2) it can be a bit painful to specify the partitioning for each
parameter.

This is where named axes come in. With Haliax, you can specify a "physical" (i.e. mesh) axis name for each
named axis in your model. Then, you just call `named_pjit` with the name of the function you want to partition, and
the `dict` containing those mappings. That's it! Here's that same example from above, but using named axes:
```python
import haliax as hax
from haliax.partitioning import named_pjit

In = hax.Axis("In", 512)
Out = hax.Axis("Out", 128)

Batch = hax.Axis("Batch", 1)

def matmul(x, y):
    return x.dot(In, y)

x = hax.ones((In, Out))
y = hax.ones((Out, Batch))

# assume we have 8 devices, e.g. a v3-8 TPU node
devices = onp.array(jax.devices()).reshape((4, 2))
mesh = Mesh(devices, ("model", "data"))

pjit_matmul = named_pjit(matmul, axis_resources={"In": "model"})

with mesh:
    print(pjit_matmul(x, y))
```

And you're done! If you want to have different partitioning for inputs and outputs, you can specify those separately:

```python
pjit_matmul = named_pjit(matmul,
                         in_axis_resources={"In": "model"},
                         out_axis_resources={"Out": "model"})
```

### Model-Partitioned GPT-2 Training

Now that we've covered the basics of `pjit` and our named variant, let's see how we can use it to train a model-parallel
GPT-2 model. XXX


## ZeRO: Parameter Partitioning

ZeRO (XXX link) is short for ZEro-Redundancy Optimizer, and it's a set of techniques for optimizing large-scale training
by partitioning model parameters, gradient accumulation buffers, and optimizer states across multiple devices, so that no
device has to hold these in memory. FSDP (short for Fully Sharded Data Parallel) is a close cousin

In GPU-land, it's more common to rely on parameter partitioning to reduce memory usage. This is because current top line
GPUs have much more memory than single TPU cores, and so it's easier to fit a larger model on a single GPU.

TPUs also claim to have much more interconnect bandwidth than GPUs, so it's
XXX this is a claim I have heard but not seen benchmarked. This from nvidia indicates hopper is 900GB/s all-reduce
while https://cloud.google.com/tpu/docs/system-architecture-tpu-vm claims 340TB/s (sic). I can't tell if this is apples to apples, but surely it's not.
https://rd.yyrcd.com/2022-03-22-NVIDIA%20Hopper%20Architecture%20In-Depth%20%7C%20NVIDIA%20Technical%20Blog.pdf



## Memory Use

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
