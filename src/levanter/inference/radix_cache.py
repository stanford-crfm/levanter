"""Radix cache for prefix lookups.

This module implements a lightweight radix tree cache that stores token
prefixes and the corresponding cache indices.  It is based on the
implementation from the `SGLang <https://github.com/lm-sys/sglang>`_
project (Apache 2.0) but stripped of all :mod:`torch` dependencies so that
it can be used inside Levanter's JAX inference stack.

Only a few methods mutate ``self``.  In particular

``reset``
    Clears the entire cache.
``match_prefix``
    Updates internal LRU timestamps when traversing the tree.
``insert``
    Adds new nodes and updates LRU state.
``evict``
    Removes least recently used nodes from the tree.
``inc_lock_ref``/``dec_lock_ref``
    Adjust lock counts on a path and update accounting of protected versus
    evictable tokens. ``lock_ref`` values mark nodes that are still in use and
    should not be removed by eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class MatchResult:
    """Result from :meth:`RadixCache.match_prefix`."""

    indices: List[int]
    last_node: "TreeNode"


class TreeNode:
    """Node in a radix tree.

    ``lock_ref`` tracks how many active references point to this node (for
    example, prefixes used by in-flight requests).  A node with
    ``lock_ref > 0`` is *protected* and will not be evicted when
    :meth:`RadixCache.evict` is called.
    """

    counter = 0

    def __init__(self, key: Optional[List[int]] = None, value: Optional[List[int]] = None):
        self.children: Dict[Tuple[int, ...], TreeNode] = {}
        self.parent: Optional[TreeNode] = None
        self.key: List[int] = key or []
        self.value: List[int] = value or []
        # number of active references that pin this node in memory
        # and protect it from eviction
        self.lock_ref: int = 0
        self.last_access_time: float = time.monotonic()
        self.id = TreeNode.counter
        TreeNode.counter += 1

    @property
    def evicted(self) -> bool:
        return self.value == []

    def __lt__(self, other: "TreeNode") -> bool:
        return self.last_access_time < other.last_access_time


class RadixCache:
    """A simple radix tree with LRU eviction."""

    def __init__(self, page_size: int = 1, disable: bool = False):
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        self.page_size = page_size
        self.disable = disable
        if self.page_size == 1:
            self._key_match_fn = self._key_match_page_size1
            self._child_key_fn = lambda k: k[0]
        else:
            self._key_match_fn = lambda k0, k1: self._key_match_paged(k0, k1)
            self._child_key_fn = lambda k: tuple(k[: self.page_size])
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the cache in place."""
        self.root_node = TreeNode([])
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: Sequence[int]) -> MatchResult:
        """Return the longest cached prefix of ``key`` and update LRU state."""
        if self.disable or not key:
            return MatchResult([], self.root_node)

        if self.page_size != 1:
            key = key[: len(key) // self.page_size * self.page_size]
        value, node = self._match_prefix_helper(self.root_node, list(key))
        return MatchResult(value, node)

    def insert(self, key: Sequence[int], value: Optional[Sequence[int]] = None) -> int:
        """Insert ``key`` into the cache in place and return matched prefix length."""
        if self.disable:
            return 0
        if value is None:
            value = list(key)
        return self._insert_helper(self.root_node, list(key), list(value))

    def evict(self, num_tokens: int) -> None:
        """Remove up to ``num_tokens`` least-recently-used tokens from the cache."""
        if self.disable:
            return
        leaves = self._collect_leaves()
        heapq.heapify(leaves)
        removed = 0
        while leaves and removed < num_tokens:
            node = heapq.heappop(leaves)
            if node is self.root_node or node.lock_ref > 0:
                continue
            removed += len(node.value)
            self._delete_leaf(node)
            if node.parent and not node.parent.children:
                heapq.heappush(leaves, node.parent)

    def inc_lock_ref(self, node: Optional[TreeNode]) -> None:
        """Increment the lock reference count for ``node`` and its parents."""
        # Nodes with a positive ``lock_ref`` are considered protected and will
        # not be removed by :meth:`evict`.  This method mutates the tree's
        # accounting of how many tokens are protected vs. evictable.
        while node and node is not self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: Optional[TreeNode]) -> None:
        """Decrement the lock reference count for ``node`` and its parents."""
        # When the lock reaches zero the node becomes eligible for eviction.
        while node and node is not self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
            node.lock_ref -= 1
            node = node.parent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _match_prefix_page(self, node: TreeNode, key: List[int]) -> Tuple[List[int], TreeNode]:
        node.last_access_time = time.monotonic()
        child_key = self._child_key_fn(key)
        value: List[int] = []
        while key and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self._key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child, prefix_len)
                value.extend(new_node.value)
                node = new_node
                break
            else:
                value.extend(child.value)
                node = child
                key = key[prefix_len:]
                if key:
                    child_key = self._child_key_fn(key)
        return value, node

    def _match_prefix_helper(self, node: TreeNode, key: List[int]) -> Tuple[List[int], TreeNode]:
        return self._match_prefix_page(node, key)

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        new_node = TreeNode(child.key[:split_len], child.value[:split_len])
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.parent = new_node
        new_node.children[self._child_key_fn(child.key)] = child
        if new_node.parent:
            new_node.parent.children[self._child_key_fn(new_node.key)] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List[int], value: List[int]) -> int:
        node.last_access_time = time.monotonic()
        if not key:
            return 0
        child_key = self._child_key_fn(key)
        total_prefix_length = 0
        while key and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self._key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            if prefix_len < len(node.key):
                node = self._split_node(node, prefix_len)
            if key:
                child_key = self._child_key_fn(key)
        if key:
            new_node = TreeNode(list(key), list(value))
            new_node.parent = node
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
        return total_prefix_length

    def _delete_leaf(self, node: TreeNode) -> None:
        parent = node.parent
        if parent is None:
            return
        for k, v in list(parent.children.items()):
            if v is node:
                del parent.children[k]
                break
        self.evictable_size_ -= len(node.value)

    def _collect_leaves(self) -> List[TreeNode]:
        leaves: List[TreeNode] = []
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            if not n.children:
                leaves.append(n)
            else:
                stack.extend(n.children.values())
        return leaves

    # ------------------------------------------------------------------
    # Key matching helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _key_match_page_size1(k0: Sequence[int], k1: Sequence[int]) -> int:
        i = 0
        for a, b in zip(k0, k1):
            if a != b:
                break
            i += 1
        return i

    def _key_match_paged(self, k0: Sequence[int], k1: Sequence[int]) -> int:
        min_len = min(len(k0), len(k1))
        i = 0
        while i < min_len:
            if list(k0[i : i + self.page_size]) != list(k1[i : i + self.page_size]):
                break
            i += self.page_size
        return i
