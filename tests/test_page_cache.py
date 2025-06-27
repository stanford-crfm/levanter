import os
import importlib.util

import equinox as eqx
import jax.numpy as jnp
import haliax as hax
from haliax import Axis

ROOT = os.path.join(os.path.dirname(__file__), "..", "src")

spec_norm = importlib.util.spec_from_file_location(
    "levanter.layers.normalization", os.path.join(ROOT, "levanter", "layers", "normalization.py")
)
norm = importlib.util.module_from_spec(spec_norm)
import sys
sys.modules[spec_norm.name] = norm
spec_norm.loader.exec_module(norm)

spec_rot = importlib.util.spec_from_file_location(
    "levanter.layers.rotary", os.path.join(ROOT, "levanter", "layers", "rotary.py")
)
rot = importlib.util.module_from_spec(spec_rot)
sys.modules[spec_rot.name] = rot
spec_rot.loader.exec_module(rot)

spec = importlib.util.spec_from_file_location(
    "levanter.layers.attention", os.path.join(ROOT, "levanter", "layers", "attention.py")
)
attention = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = attention
spec.loader.exec_module(attention)

PageCache = attention.PageCache


def test_page_cache_extend_simple():
    Seq = Axis("seq", 2)
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

    assert jnp.all(cache.kv_lens.array == jnp.array([1, 1], dtype=jnp.int32))
    assert jnp.all(cache.page_indices.array == jnp.array([[0, -1], [1, -1]], dtype=jnp.int32))
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
    Seq = Axis("seq", 1)
    Page = Axis("page", 2)
    MaxPage = Axis("max_page", 2)
    Slot = Axis("slot", 2)
    KVH = Axis("kv_head", 1)
    HD = Axis("head_dim", 1)

    cache = PageCache.init(Seq, Page, Slot, KVH, HD, MaxPage, dtype=jnp.float32)

    Tok1 = Axis("tok1", 1)
    k1 = hax.ones((Tok1, KVH, HD), dtype=jnp.float32)
    v1 = hax.ones((Tok1, KVH, HD), dtype=jnp.float32) * 100
    cu1 = jnp.array([0, 1], dtype=jnp.int32)
    pages1 = jnp.array([0], dtype=jnp.int32)
    cache = PageCache.extend(cache, k1, v1, cu1, pages1, 1)

    Tok2 = Axis("tok2", 2)
    k2 = hax.arange(Tok2).broadcast_axis((KVH, HD)).rearrange((Tok2, KVH, HD)) + 2
    v2 = hax.arange(Tok2).broadcast_axis((KVH, HD)).rearrange((Tok2, KVH, HD)) + 102
    cu2 = jnp.array([0, 2], dtype=jnp.int32)
    pages2 = jnp.array([1], dtype=jnp.int32)
    cache = PageCache.extend(cache, k2, v2, cu2, pages2, 1)

    assert cache.kv_lens.array[0] == 3
    assert jnp.all(cache.page_indices.array[0] == jnp.array([0, 1], dtype=jnp.int32))
    assert cache.kv_pages.array[0, 0, 0, 0] == 1
    assert cache.kv_pages.array[0, 1, 0, 0] == 2
    assert cache.kv_pages.array[1, 0, 0, 0] == 3
