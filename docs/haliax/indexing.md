# Indexing in Haliax

Haliax supports Numpy-style indexing, including advanced indexing, though the syntax is necessarily different.
Most forms of indexing are supporting, except we don't support indexing with booleans right now. (JAX doesn't support indexing with non-constant bool arrays anyway,
so I don't think it's worth the effort to implement it in Haliax.)

## Basic Indexing

Basic indexing works basically like you would expect: you can use integers or slices to index into an array.
Haliax supports two syntaxes for indexing: one accepts a dict of axis names and indices, and the other accepts
an alternating sequence of axis names and indices. The latter is useful for indexing with a small number of indices.


```python
import haliax as hax
import jax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))

a[{"X": 1, "Y": 2, "Z": 3}]  # returns a scalar jnp.ndarray
a[{"X": 1, "Y": 2, "Z": slice(3, 5)}]  # return a NamedArray with axes = Axis("Z", 2)
a[{"X": 1, "Y": slice(2, 4), "Z": slice(3, 5)}]  # return a NamedArray with axes = Axis("Y", 2), Axis("Z", 2)

a["X", 1, "Y", 2, "Z", 3]  # returns a scalar jnp.ndarray
a["X", 1, "Y", 2, "Z", 3:5]  # return a NamedArray with axes = Axis("Z", 2)
a["X", 1, "Y", 2:4, "Z", 3:5]  # return a NamedArray with axes = Axis("Y", 2), Axis("Z", 2)
```

Unfortunately, Python won't let us use `:` slice syntax inside of a dictionary, so we have to use `slice` instead.
This is why we have the second syntax, which is a bit less idiomatic in some ways, but it's more convenient.

Otherwise, the idea is pretty straightforward: any unspecified axes are treated as though indexed with `:` in NumPy,
slices are kept in reduced dimensions, and integers eliminate dimensions. If all dimensions are eliminated, a scalar
JAX ndarray is returned.

## Advanced Indexing

NumPy's [Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing) is supported,
though we use named arrays for the indices instead of normal arrays. In NumPy, the indexed arrays much be
broadcastable to the same shape. Advanced indexing in Haliax is similar, except that they follow Haliax's broadcasting rules,
meaning that the axis names determine broadcasting. Axes with the same name must have the same size.

```python
import haliax as hax
import jax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))

I1 = hax.Axis("I1", 5)
I2 = hax.Axis("I2", 5)
I3 = hax.Axis("I3", 5)
ind1 = hax.random.randint(jax.random.PRNGKey(0), (I1,), 0, 10)
ind2 = hax.random.randint(jax.random.PRNGKey(0), (I2, I3), 0, 20)

a[{"X": ind1}]  # returns a NamedArray with axes = Axis("I1", 5), Axis("Y", 20), Axis("Z", 30)

a[{"X": ind1, "Y": ind2}]  # returns a NamedArray with axes = Axis("I1", 5), Axis("I2", 5), Axis("I3", 5), Axis("Z", 30)
a[{"X": ind1, "Y": ind2, "Z": 3}]  # returns a NamedArray with axes = Axis("I1", 5), Axis("I2", 5), Axis("I3", 5)
```

The order of the indices in the dictionary doesn't matter, and you can mix and match basic and advanced indexing.
The actual sequence of axes is a bit complex, both in Haliax and in NumPy. If you need a specific order, it's probably
best to use rearrange.

In keeping with the one-axis-per-name rule, you are allowed to index using axes with a name present in the array,
if it would be eliminated by the indexing operation. For example:

```python
import haliax as hax
import jax

X = hax.Axis("X", 10)
Y = hax.Axis("Y", 20)
Z = hax.Axis("Z", 30)

X2 = hax.Axis("X", 5)
Y2 = hax.Axis("Y", 5)

a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))
ind1 = hax.random.randint(jax.random.PRNGKey(0), (X2,), 0, 10)
ind2 = hax.random.randint(jax.random.PRNGKey(0), (Y2,), 0, 10)

a[{"X": ind1, "Y": ind2}]  # returns a NamedArray with axes = Axis("X", 5), Axis("Y", 5), Axis("Z", 30)

a[{"Y": ind1}]  # error, "X" is not eliminated by the indexing operation

a[{"X": ind2, "Y": ind1}]  # ok, because X and Y are eliminated by the indexing operation
```
