import importlib.util
import sys
from pathlib import Path

# Load the module directly so we avoid running ``levanter.__init__`` which
# pulls in heavy dependencies.
rc_path = Path(__file__).resolve().parents[1] / "src/levanter/inference/radix_cache.py"
spec = importlib.util.spec_from_file_location("radix_cache", rc_path)
assert spec is not None
radix_cache = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = radix_cache
assert spec.loader is not None
spec.loader.exec_module(radix_cache)
RadixCache = radix_cache.RadixCache


def test_insert_and_match():
    cache = RadixCache(page_size=1)
    cache.insert([1, 2, 3], [10, 20, 30])
    res = cache.match_prefix([1, 2, 3, 4])
    assert res.indices == [10, 20, 30]
    assert res.last_node.key == [1, 2, 3]


def test_partial_match_and_split():
    cache = RadixCache(page_size=1)
    cache.insert([1, 2, 3], [10, 20, 30])
    res = cache.match_prefix([1, 2, 4, 5])
    assert res.indices == [10, 20]
    assert res.last_node.key == [1, 2]


def test_lru_eviction():
    cache = RadixCache(page_size=1)
    cache.insert([1], [1])
    cache.insert([2], [2])
    cache.insert([3], [3])

    # Access node [3] and [1] to make [2] least recently used
    cache.match_prefix([1])
    cache.match_prefix([3])

    cache.evict(1)

    assert cache.match_prefix([2]).indices == []
    assert cache.match_prefix([1]).indices == [1]
    assert cache.match_prefix([3]).indices == [3]


def test_pagesize_two_and_split():
    cache = RadixCache(page_size=2)
    cache.insert([1, 2, 3, 4])

    res = cache.match_prefix([1, 2, 5, 6, 7])
    assert res.indices == [1, 2]
    assert res.last_node.key == [1, 2]

    cache.insert([1, 2, 5, 6])
    res = cache.match_prefix([1, 2, 5, 6, 7])
    assert res.indices == [1, 2, 5, 6]
    assert res.last_node.key == [5, 6]


def test_pagesize_two_almost_common_prefix():
    cache = RadixCache(page_size=2)
    cache.insert([1, 2, 3, 4])

    # first page differs so no prefix should match
    res = cache.match_prefix([1, 3, 5, 6])
    assert res.indices == []
    assert res.last_node is cache.root_node

    cache.insert([1, 3, 5, 6])
    res = cache.match_prefix([1, 3, 5, 6, 7])
    assert res.indices == [1, 3, 5, 6]
    assert res.last_node.key == [1, 3, 5, 6]


def test_lock_ref_protection():
    cache = RadixCache(page_size=1)
    cache.insert([1, 2, 3])
    cache.insert([4])
    node = cache.match_prefix([1, 2, 3]).last_node

    cache.inc_lock_ref(node)
    cache.evict(10)
    # node [1,2,3] should still be present
    assert cache.match_prefix([1, 2, 3]).indices == [1, 2, 3]
    cache.dec_lock_ref(node)
    cache.evict(10)
    # Now it can be evicted
    assert cache.match_prefix([1, 2, 3]).indices == []
