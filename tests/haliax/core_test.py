import jax
import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis, NamedArray


def test_dot():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot(Height, m1, m2).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            hax.dot((Height, Width), m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.dot((Height, Width, Depth), m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_dot_string_selection():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))
    m2 = NamedArray(jnp.ones((Depth.size, Width.size, Height.size)), (Depth, Width, Height))

    assert jnp.all(jnp.equal(hax.dot("Height", m1, m2).array, jnp.einsum("ijk,kji->jk", m1.array, m2.array)))
    assert jnp.all(
        jnp.equal(
            hax.dot(("Height", "Width"), m1, m2).array,
            jnp.einsum("ijk,kji->k", m1.array, m2.array),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.dot(("Height", "Width", "Depth"), m1, m2).array,
            jnp.einsum("ijk,kji->", m1.array, m2.array),
        )
    )


def test_unary_np_functions():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    m1 = NamedArray(jnp.ones((Height.size, Width.size, Depth.size)), (Height, Width, Depth))

    assert jnp.all(jnp.equal(hax.abs(m1).array, jnp.abs(m1.array)))
    assert jnp.all(jnp.equal(hax.absolute(m1).array, jnp.absolute(m1.array)))
    assert jnp.all(jnp.equal(hax.angle(m1).array, jnp.angle(m1.array)))
    assert jnp.all(jnp.equal(hax.arccos(m1).array, jnp.arccos(m1.array)))
    assert jnp.all(jnp.equal(hax.arccosh(m1).array, jnp.arccosh(m1.array)))
    assert jnp.all(jnp.equal(hax.arcsin(m1).array, jnp.arcsin(m1.array)))
    assert jnp.all(jnp.equal(hax.arcsinh(m1).array, jnp.arcsinh(m1.array)))
    assert jnp.all(jnp.equal(hax.arctan(m1).array, jnp.arctan(m1.array)))
    assert jnp.all(jnp.equal(hax.arctanh(m1).array, jnp.arctanh(m1.array)))
    assert jnp.all(jnp.equal(hax.around(m1).array, jnp.around(m1.array)))
    assert jnp.all(jnp.equal(hax.cbrt(m1).array, jnp.cbrt(m1.array)))


def test_reduction_functions():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    rand_m = jax.random.uniform(PRNGKey(0), (Height.size, Width.size, Depth.size))

    m1 = NamedArray(rand_m, (Height, Width, Depth))

    # sum out everything
    assert jnp.all(jnp.equal(hax.sum(m1).array, jnp.sum(m1.array)))
    # ensure it's a scalar

    assert jnp.all(jnp.equal(hax.sum(m1, axis=Height).array, jnp.sum(m1.array, axis=0)))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=Width).array, jnp.sum(m1.array, axis=1)))

    # sum out two axes
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Height, Width)).array, jnp.sum(m1.array, axis=(0, 1))))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Width, Height)).array, jnp.sum(m1.array, axis=(1, 0))))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(Height, Depth)).array, jnp.sum(m1.array, axis=(0, 2))))

    # sum out three axes
    assert jnp.all(
        jnp.equal(
            hax.sum(m1, axis=(Height, Width, Depth)).array,
            jnp.sum(m1.array, axis=(0, 1, 2)),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.sum(m1, axis=(Width, Height, Depth)).array,
            jnp.sum(m1.array, axis=(1, 0, 2)),
        )
    )

    # argmax
    assert jnp.all(jnp.equal(hax.argmax(m1).array, jnp.argmax(m1.array)))
    assert jnp.all(jnp.equal(hax.argmax(m1, axis=Height).array, jnp.argmax(m1.array, axis=0)))


def test_reduction_functions_with_where():
    H = Axis("H", 2)
    W = Axis("W", 3)
    D = Axis("D", 4)

    rand_m = jax.random.uniform(PRNGKey(0), (H.size, W.size, D.size))

    m1 = NamedArray(rand_m, (H, W, D))

    mask = m1 > 0.5
    jmask = m1.array > 0.5

    # sum out everything
    assert jnp.all(jnp.equal(hax.sum(m1, where=mask).array, jnp.sum(rand_m, where=jmask)))
    # ensure it's a scalar

    assert jnp.all(jnp.equal(hax.sum(m1, axis=H, where=mask).array, jnp.sum(rand_m, axis=0, where=jmask)))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=W, where=mask).array, jnp.sum(rand_m, axis=1, where=jmask)))

    assert jnp.all(jnp.equal(hax.sum(m1, axis="H", where=mask).array, jnp.sum(rand_m, axis=0, where=jmask)))

    # sum out two axes
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(H, W), where=mask).array, jnp.sum(rand_m, axis=(0, 1), where=jmask)))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(W, H), where=mask).array, jnp.sum(rand_m, axis=(1, 0), where=jmask)))
    assert jnp.all(jnp.equal(hax.sum(m1, axis=(H, D), where=mask).array, jnp.sum(rand_m, axis=(0, 2), where=jmask)))

    assert jnp.all(
        jnp.equal(hax.sum(m1, axis=("H", "W"), where=mask).array, jnp.sum(rand_m, axis=(0, 1), where=jmask))
    )

    # sum out three axes
    assert jnp.all(
        jnp.equal(
            hax.sum(m1, axis=(H, W, D), where=mask).array,
            jnp.sum(rand_m, axis=(0, 1, 2), where=jmask),
        )
    )
    assert jnp.all(
        jnp.equal(
            hax.sum(m1, axis=(W, H, D), where=mask).array,
            jnp.sum(rand_m, axis=(1, 0, 2), where=jmask),
        )
    )

    assert jnp.all(
        jnp.equal(hax.sum(m1, axis=("H", "W", "D"), where=mask).array, jnp.sum(rand_m, axis=(0, 1, 2), where=jmask))
    )


def test_split():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    D10 = Axis("Depth", Depth.size * 10)

    rand_m = hax.random.uniform(PRNGKey(0), (Height, Width, D10))
    m = rand_m.array

    splits = hax.split(rand_m, axis=D10, new_axes=[Depth] * 10)

    assert splits[0].axes == (Height, Width, Depth)
    assert len(splits) == 10

    usplits = jnp.split(m, 10, axis=2)

    for i in range(10):
        assert jnp.all(jnp.equal(splits[i].array, usplits[i]))

    # double check string axis
    splits_str = hax.split(rand_m, axis="Depth", new_axes=[Depth] * 10)
    for i in range(10):
        assert jnp.all(jnp.equal(splits_str[i].array, usplits[i]))


def test_take():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)
    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    assert jnp.all(jnp.equal(hax.take(named1, Height, 0).array, named1.array[0]))

    Index = Axis("Index", 5)
    indices = hax.ones(Index, dtype=jnp.int32)

    named2 = hax.take(named1, Height, indices)
    assert named2.axes == (Index, Width, Depth)

    named2 = hax.take(named1, "Width", indices)
    assert named2.axes == (Height, Index, Depth)

    named2 = hax.take(named1, Depth, indices)
    assert named2.axes == (Height, Width, Index)

    Index2 = Axis("Index2", 3)

    indices2 = hax.ones((Index, Index2), dtype=jnp.int32)

    named2 = hax.take(named1, Height, indices2)
    assert named2.axes == (Index, Index2, Width, Depth)

    named2 = hax.take(named1, "Width", indices2)
    assert named2.axes == (Height, Index, Index2, Depth)

    named2 = hax.take(named1, Depth, indices2)
    assert named2.axes == (Height, Width, Index, Index2)


def test_cumsum_etc():
    Height = Axis("Height", 2)
    Width = Axis("Width", 3)
    Depth = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (Height, Width, Depth))

    assert jnp.all(jnp.equal(hax.cumsum(named1, axis=Height).array, jnp.cumsum(named1.array, axis=0)))
    assert hax.cumsum(named1, axis=Height).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumsum(named1, axis=Width).array, jnp.cumsum(named1.array, axis=1)))
    assert hax.cumsum(named1, axis=Width).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumsum(named1, axis=Depth).array, jnp.cumsum(named1.array, axis=2)))
    assert hax.cumsum(named1, axis=Depth).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumprod(named1, axis=Height).array, jnp.cumprod(named1.array, axis=0)))
    assert hax.cumprod(named1, axis=Height).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumprod(named1, axis=Width).array, jnp.cumprod(named1.array, axis=1)))
    assert hax.cumprod(named1, axis=Width).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.cumprod(named1, axis=Depth).array, jnp.cumprod(named1.array, axis=2)))
    assert hax.cumprod(named1, axis=Depth).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.argsort(named1, axis=Height).array, jnp.argsort(named1.array, axis=0)))
    assert hax.argsort(named1, axis=Height).axes == (Height, Width, Depth)

    assert jnp.all(jnp.equal(hax.argsort(named1, axis=Width).array, jnp.argsort(named1.array, axis=1)))
    assert hax.argsort(named1, axis=Width).axes == (Height, Width, Depth)


def test_rearrange():
    H = Axis("H", 2)
    W = Axis("W", 3)
    D = Axis("D", 4)
    C = Axis("C", 5)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D, C))

    assert jnp.all(jnp.equal(hax.rearrange(named1, (C, W, D, H)).array, jnp.transpose(named1.array, (3, 1, 2, 0))))
    assert hax.rearrange(named1, (C, W, D, H)).axes == (C, W, D, H)

    # test str args
    assert jnp.all(
        jnp.equal(hax.rearrange(named1, ("C", "W", "D", "H")).array, jnp.transpose(named1.array, (3, 1, 2, 0)))
    )
    assert hax.rearrange(named1, ("C", "W", "D", "H")).axes == (C, W, D, H)
    # test mixed str and Axis args
    assert jnp.all(jnp.equal(hax.rearrange(named1, ("C", W, "D", H)).array, jnp.transpose(named1.array, (3, 1, 2, 0))))

    # test ellipsis
    assert jnp.all(jnp.equal(hax.rearrange(named1, (C, ..., D)).array, jnp.transpose(named1.array, (3, 0, 1, 2))))

    # test errors for double ellipsis
    with pytest.raises(ValueError):
        hax.rearrange(named1, (C, ..., ...))

    # test errors for multiply specified axes
    with pytest.raises(ValueError):
        hax.rearrange(named1, (C, W, W, H))

    # test errors for unknown axes
    with pytest.raises(ValueError):
        X = Axis("X", 6)
        hax.rearrange(named1, (C, X, D, H))

    # test for missing axes
    with pytest.raises(ValueError):
        hax.rearrange(named1, (C, W, D))


def test_rearrange_unused_ellipsis():
    # Make sure we just ignore the ellipsis if all axes are specified in addition
    H = Axis("Height", 2)
    W = Axis("Width", 3)
    D = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))

    assert jnp.all(jnp.equal(hax.rearrange(named1, (H, W, D, ...)).array, named1.array))
    assert hax.rearrange(named1, (H, W, D, ...)).axes == (H, W, D)

    assert jnp.all(jnp.equal(hax.rearrange(named1, (H, ..., W, D)).array, named1.array))
    assert hax.rearrange(named1, (H, ..., W, D)).axes == (H, W, D)

    assert jnp.all(jnp.equal(hax.rearrange(named1, (D, ..., W, H)).array, jnp.transpose(named1.array, (2, 1, 0))))
    assert hax.rearrange(named1, (D, ..., W, H)).axes == (D, W, H)


def test_arange():
    H = Axis("Height", 10)

    assert jnp.all(jnp.equal(hax.arange(H).array, jnp.arange(10)))
    assert hax.arange(H).axes == (H,)

    # test stride
    assert jnp.all(jnp.equal(hax.arange(H, step=2).array, jnp.arange(0, 20, 2)))

    # test start and stride
    assert jnp.all(jnp.equal(hax.arange(H, start=2, step=2).array, jnp.arange(2, 22, 2)))


def test_stack():
    H = Axis("H", 4)
    W = Axis("W", 3)

    named1 = hax.random.uniform(PRNGKey(0), (H, W))
    named2 = hax.random.uniform(PRNGKey(1), (H, W))

    assert jnp.all(jnp.equal(hax.stack("B", (named1, named2)).array, jnp.stack((named1.array, named2.array), axis=0)))

    named3 = hax.random.uniform(PRNGKey(2), (W, H))
    # test that this rearranges fine
    reord_stack = hax.stack("B", (named1, named3))
    assert jnp.all(jnp.equal(reord_stack.array, jnp.stack((named1.array, named3.array.transpose(1, 0)), axis=0)))
    assert reord_stack.axes == (Axis("B", 2), H, W)


def test_concatenate():
    H1 = Axis("H", 4)
    H2 = Axis("H", 3)
    W = Axis("W", 3)

    named1 = hax.random.uniform(PRNGKey(0), (H1, W))
    named2 = hax.random.uniform(PRNGKey(1), (H2, W))

    assert jnp.all(
        jnp.equal(hax.concatenate("H", (named1, named2)).array, jnp.concatenate((named1.array, named2.array), axis=0))
    )
    assert hax.concatenate("H", (named1, named2)).axes == (Axis("H", 7), W)

    # test that this rearranges fine
    named3 = hax.random.uniform(PRNGKey(2), (W, H2))
    reord_concat = hax.concatenate("H", (named1, named3))
    assert jnp.all(
        jnp.equal(reord_concat.array, jnp.concatenate((named1.array, named3.array.transpose(1, 0)), axis=0))
    )

    # test we can concatenate along the 2nd axis
    named1 = named1.rearrange((W, H1))
    named2 = named2.rearrange((W, H2))

    assert jnp.all(
        jnp.equal(hax.concatenate("H", (named1, named2)).array, jnp.concatenate((named1.array, named2.array), axis=1))
    )
    assert hax.concatenate("H", (named1, named2)).axes == (W, Axis("H", 7))


def test_unflatten_axis():
    H = Axis("Height", 2)
    W = Axis("Width", 3)
    D = Axis("Depth", 4)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))
    flattened_HW = named1.flatten_axes((H, W), "Z")

    assert jnp.all(jnp.equal(hax.unflatten_axis(flattened_HW, "Z", (H, W)).array, named1.array))
    assert hax.unflatten_axis(flattened_HW, "Z", (H, W)).axes == (H, W, D)

    assert jnp.all(jnp.equal(hax.unflatten_axis(flattened_HW, "Z", (H, W)).array, named1.array))

    # test that we can unflatten to a different order
    # in general, this won't be equivalent to the original array
    assert not jnp.all(jnp.equal(hax.unflatten_axis(flattened_HW, "Z", (W, H)).array, named1.array.transpose(1, 0, 2)))
    assert hax.unflatten_axis(flattened_HW, "Z", (W, H)).axes == (W, H, D)

    # flatten non-consecutive axes
    flattened_HD = named1.flatten_axes((H, D), "Z")
    assert jnp.all(jnp.equal(hax.unflatten_axis(flattened_HD, "Z", (H, D)).array, named1.array.transpose(0, 2, 1)))
    assert hax.unflatten_axis(flattened_HD, "Z", (H, D)).axes == (H, D, W)


def test_rename():
    H = Axis("H", 2)
    W = Axis("W", 3)
    D = Axis("D", 4)

    H2 = Axis("H2", 2)
    W2 = Axis("W2", 3)
    D2 = Axis("D2", 4)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))

    assert jnp.all(jnp.equal(hax.rename(named1, {"H": "H2", "W": "W2", "D": "D2"}).array, named1.array))
    assert hax.rename(named1, {"H": "H2", "W": "W2", "D": "D2"}).axes == (H2, W2, D2)

    assert jnp.all(jnp.equal(hax.rename(named1, {"H": H2, "W": "W2"}).array, named1.array))
    assert hax.rename(named1, {"H": H2, "W": "W2"}).axes == (H2, W2, D)

    assert jnp.all(jnp.equal(hax.rename(named1, {H: H2, "W": "W2"}).array, named1.array))
    assert hax.rename(named1, {H: H2, "W": "W2"}).axes == (H2, W2, D)


def test_slice_nd():
    H = Axis("H", 20)
    W = Axis("W", 30)
    D = Axis("D", 40)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))

    assert jnp.all(jnp.equal(hax.slice_nd(named1, {"H": slice(0, 10, 2)}).array, named1.array[0:10:2, :, :]))
    assert hax.slice_nd(named1, {"H": slice(0, 10, 2)}).axes == (Axis("H", 5), W, D)

    # try indexing syntax
    assert jnp.all(jnp.equal(named1[{"H": slice(0, 10, 2)}].array, named1.array[0:10:2, :, :]))
    assert named1[{"H": slice(0, 10, 2)}].axes == (Axis("H", 5), W, D)

    # try indexing syntax with multiple slices
    assert jnp.all(
        jnp.equal(named1[{"H": slice(3, 13, 2), "W": slice(0, 10, 2)}].array, named1.array[3:13:2, 0:10:2, :])
    )

    # try indexing with 1 slice and 1 integer
    assert jnp.all(jnp.equal(named1[{"H": slice(0, 10, 2), "W": 0}].array, named1.array[0:10:2, 0, :]))
    assert named1[{"H": slice(0, 10, 2), "W": 0}].axes == (Axis("H", 5), D)

    # try indexing with 3 integers: returns scalar ndarray
    assert jnp.all(jnp.equal(named1[{"H": 0, "W": 0, "D": 0}], named1.array[0, 0, 0]))


def test_slice_nd_array_slices():
    # fancier tests with array slices with named array args
    H = Axis("H", 10)
    W = Axis("W", 20)
    D = Axis("D", 30)
    C = Axis("C", 40)
    Q = Axis("Q", 50)
    I0 = Axis("I0", 10)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D, C, Q))
    index_1 = hax.random.randint(PRNGKey(0), (I0,), 0, H.size)

    assert jnp.all(jnp.equal(named1[{"H": index_1}].array, named1.array[index_1.array, :, :]))
    assert named1[{"H": index_1}].axes == (I0, W, D, C, Q)

    # try indexing with 1 array and 1 integer
    assert jnp.all(jnp.equal(named1[{"H": index_1, "W": 0}].array, named1.array[index_1.array, 0, :]))
    assert named1[{"H": index_1, "W": 0}].axes == (I0, D, C, Q)

    # more complex case: advanced indices aren't contiguous
    assert jnp.all(jnp.equal(named1[{"H": index_1, "D": 0}].array, named1.array[index_1.array, :, 0]))
    assert named1[{"H": index_1, "D": 0}].axes == (I0, W, C, Q)

    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # Example
    # Let x.shape be (10, 20, 30, 40, 50) and suppose ind_1 and ind_2 can be broadcast to the shape (2, 3, 4).
    I1 = Axis("I1", 2)
    I2 = Axis("I2", 3)
    I3 = Axis("I3", 4)

    ind_1 = hax.random.randint(PRNGKey(0), (I2, I3), 0, W.size)
    ind_2 = hax.random.randint(PRNGKey(0), (I1, I3), 0, D.size)

    # Then x[:, ind_1, ind_2] has shape (10, 2, 3, 4, 40, 50) because the (20, 30)-shaped subspace from X has been replaced with the (2, 3, 4) subspace from the indices.
    assert jnp.all(
        jnp.equal(
            named1[{"W": ind_1, "D": ind_2}].array,
            named1.array[:, ind_1.array.reshape(1, 3, 4), ind_2.array.reshape(2, 1, 4), :],
        )
    )
    assert named1[{"W": ind_1, "D": ind_2}].axes == (H, I1, I2, I3, C, Q)

    # However, x[:, ind_1, :, ind_2] has shape (2, 3, 4, 10, 30, 50) because there is no unambiguous place to drop in the indexing subspace, thus it is tacked-on to the beginning. It is always possible to use .transpose() to move the subspace anywhere desired. Note that this example cannot be replicated using take.
    assert jnp.all(
        jnp.equal(
            named1[{"W": ind_1, "C": ind_2}].array,
            named1.array[:, ind_1.array.reshape(1, 3, 4), :, ind_2.array.reshape(2, 1, 4), :],
        )
    )
    assert named1[{"W": ind_1, "C": ind_2}].axes == (I1, I2, I3, H, D, Q)


def test_slice_nd_shorthand_syntax():
    # syntax like arr["X", 0:10, "Y", 0:10] is supported

    H = Axis("H", 10)
    W = Axis("W", 20)
    D = Axis("D", 30)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))

    assert jnp.all(jnp.equal(named1["H", 0:10, "D", 0:10].array, named1.array[0:10, :, 0:10]))


def test_slice_nd_array_present_dims():
    # tests slicing with arrays that are already present in the named array, which is sometimes ok
    H = Axis("H", 10)
    W = Axis("W", 20)
    D = Axis("D", 30)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))

    index1 = hax.random.randint(PRNGKey(0), (H,), 0, H.size)

    # this is ok, since the H would be eliminated anyway
    assert jnp.all(jnp.equal(named1[{"H": index1}].array, named1.array[index1.array, :, :]))

    # this is not ok, since the H would not be eliminated
    with pytest.raises(ValueError):
        named1[{W: index1}]

    # this is not ok, but is trickier because the H has a different size
    H2 = H.resize(5)
    index2 = hax.random.randint(PRNGKey(0), (H2,), 0, H.size)
    with pytest.raises(ValueError):
        named1[{W: index2}]

    # this is ok, since the H would be eliminated anyway
    assert jnp.all(jnp.equal(named1[{"H": index2}].array, named1.array[index2.array, :, :]))


def test_full_indexing_returns_scalar():
    H = Axis("H", 10)
    W = Axis("W", 20)
    D = Axis("D", 30)

    named1 = hax.random.uniform(PRNGKey(0), (H, W, D))
    sliced = named1[{"H": 0, "W": 0, "D": 0}]

    assert isinstance(sliced, jnp.ndarray)
    assert sliced.shape == ()


def test_indexing_bug_from_docs():
    X = hax.Axis("X", 10)
    Y = hax.Axis("Y", 20)
    Z = hax.Axis("Z", 30)

    a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))

    I1 = hax.Axis("I1", 5)
    I2 = hax.Axis("I2", 5)
    I3 = hax.Axis("I3", 5)
    ind1 = hax.random.randint(jax.random.PRNGKey(0), (I1,), 0, 10)
    ind2 = hax.random.randint(jax.random.PRNGKey(0), (I2, I3), 0, 20)

    # assert a[{"X": ind1, "Y": ind2}].axes == (I1, I2, I3, Z)
    assert a[{"X": ind1, "Y": ind2, "Z": 3}].axes == (I1, I2, I3)


def test_duplicate_axis_names_in_slicing():
    X = hax.Axis("X", 10)
    Y = hax.Axis("Y", 20)
    Z = hax.Axis("Z", 30)

    X2 = hax.Axis("X", 5)
    Y2 = hax.Axis("Y", 5)

    a = hax.random.uniform(jax.random.PRNGKey(0), (X, Y, Z))
    ind1 = hax.random.randint(jax.random.PRNGKey(0), (X2,), 0, 10)
    ind2 = hax.random.randint(jax.random.PRNGKey(0), (Y2,), 0, 10)

    a[{"X": ind1, "Y": ind2}]  # returns a NamedArray with axes = Axis("X", 5), Axis("Y", 5), Axis("Z", 30)

    with pytest.raises(ValueError):
        a[{"Y": ind1}]  # error, "X" is not eliminated by the indexing operation

    a[{"X": ind2, "Y": ind1}]  # ok, because X and Y are eliminated by the indexing operation
