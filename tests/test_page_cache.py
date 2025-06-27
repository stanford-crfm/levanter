import equinox as eqx
import jax.numpy as jnp

import haliax as hax
from haliax import Axis

from levanter.layers.attention import PageCache


def test_page_cache_extend_simple():
    Seq = Axis("seq", 4)
    Page = Axis("page", 2)
    MaxPage = Axis("max_page", 2)
    Slot = Axis("slot", 2)
    KVH = Axis("kv_head", 1)
    HD = Axis("head_dim", 1)

    cache = PageCache.init(Seq, Page, Slot, KVH, HD, MaxPage, dtype=jnp.float32)

    Tok = Axis("tok", 2)
    new_k = hax.arange(Tok).broadcast_axis((KVH, HD)).rearrange((Tok, KVH, HD)) + 1
    new_v = hax.arange(Tok).broadcast_axis((KVH, HD)).rearrange((Tok, KVH, HD)) + 101

    cu = jnp.array([0, 1, 2], dtype=jnp.int32)
    pages = jnp.array([0, 1], dtype=jnp.int32)

    jit_extend = eqx.filter_jit(PageCache.extend)
    cache = jit_extend(cache, new_k, new_v, cu, pages, 2)

    assert jnp.all(cache.kv_lens.array == jnp.array([1, 1, 0, 0], dtype=jnp.int32))
    assert jnp.all(cache.page_indices.array == jnp.array([[0, -1], [1, -1], [-1, -1], [-1, -1]], dtype=jnp.int32))
    assert cache.kv_pages.array[0, 0, 0, 0] == 1
    assert cache.kv_pages.array[0, 0, 1, 0] == 101
    assert cache.kv_pages.array[1, 0, 0, 0] == 2
    assert cache.kv_pages.array[1, 0, 1, 0] == 102


def test_page_cache_extend_multi_page():
    Seq = Axis("seq", 2)
    Page = Axis("page", 3)
    MaxPage = Axis("max_page", 3)
    Slot = Axis("slot", 2)
    KVH = Axis("kv_head", 1)
    HD = Axis("head_dim", 1)

    cache = PageCache.init(Seq, Page, Slot, KVH, HD, MaxPage, dtype=jnp.float32)

    Tok = Axis("tok", 4)
    new_k = hax.arange(Tok).broadcast_axis((KVH, HD)).rearrange((Tok, KVH, HD)) + 1
    new_v = hax.arange(Tok).broadcast_axis((KVH, HD)).rearrange((Tok, KVH, HD)) + 101

    cu = jnp.array([0, 3, 4], dtype=jnp.int32)
    pages = jnp.array([0, 1, 2], dtype=jnp.int32)

    jit_extend = eqx.filter_jit(PageCache.extend)
    cache = jit_extend(cache, new_k, new_v, cu, pages, 2)

    assert jnp.all(cache.kv_lens.array == jnp.array([3, 1], dtype=jnp.int32))
    assert cache.page_indices.array[0, 0] == 0
    assert cache.page_indices.array[0, 1] == 1
    assert cache.page_indices.array[1, 0] == 2

    assert cache.kv_pages.array[0, 0, 0, 0] == 1
    assert cache.kv_pages.array[0, 1, 0, 0] == 2
    assert cache.kv_pages.array[1, 0, 0, 0] == 3
    assert cache.kv_pages.array[2, 0, 0, 0] == 4

    assert cache.kv_pages.array[0, 0, 1, 0] == 101
    assert cache.kv_pages.array[0, 1, 1, 0] == 102
    assert cache.kv_pages.array[1, 0, 1, 0] == 103
    assert cache.kv_pages.array[2, 0, 1, 0] == 104


def test_page_cache_partial_page_extension():
    Seq = Axis("seq", 2)
    Page = Axis("page", 2)
    MaxPage = Page.resize(4)
    Slot = Axis("slot", 2)
    KVH = Axis("kv_head", 1)
    HD = Axis("head_dim", 1)
    Pos = Axis("pos", 16)

    cache = PageCache.init(Seq, Page, Slot, KVH, HD, MaxPage, dtype=jnp.float32)

    k_base = hax.full((Pos, KVH, HD), -1, dtype=jnp.float32)
    v_base = hax.full((Pos, KVH, HD), -1, dtype=jnp.float32)
    k1 = k_base.at["pos", 0].set(1)
    v1 = v_base.at["pos", 0].set(100)
    cu1 = jnp.array([0, 1], dtype=jnp.int32)
    pages1 = jnp.zeros((1,), jnp.int32)
    cache = PageCache.extend(cache, k1, v1, cu1, pages1, 1)

    k2 = k_base.at["pos", 0:2].set(hax.arange(Pos.resize(2), start=2))
    v2 = v_base.at["pos", 0:2].set(hax.arange(Pos.resize(2), start=101))
    cu2 = jnp.array([0, 2, 2], dtype=jnp.int32)
    pages2 = jnp.array([1], dtype=jnp.int32)
    cache = PageCache.extend(cache, k2, v2, cu2, pages2, 1)

    assert jnp.all(cache.kv_lens.array == jnp.array([3, 0], dtype=jnp.int32))
    assert cache.page_indices.array[0, 0] == 0
    assert cache.page_indices.array[0, 1] == 1
    assert jnp.all(cache.page_indices.array[0, 2:] == -1)
    assert jnp.all(cache.page_indices.array[1, :] == -1)
