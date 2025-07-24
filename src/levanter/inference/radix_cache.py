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

    ``is_partial`` indicates that the node represents an incomplete page
    (``len(key) < page_size``).  Such nodes can later be extended when more
    tokens become available.
    """

    counter = 0

    def __init__(
        self,
        key: Optional[List[int]] = None,
        value: Optional[List[int]] = None,
        *,
        is_partial: bool = False,
    ):
        self.children: Dict[Tuple[int, ...], TreeNode] = {}
        self.parent: Optional[TreeNode] = None
        self.key: List[int] = key or []
        self.value: List[int] = value or []
        self.is_partial = is_partial
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
        self._key_match_fn = self._key_match_seq
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the cache in place."""
        self.root_node = TreeNode([], is_partial=False)
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def match_prefix(self, key: Sequence[int]) -> MatchResult:
        """Return the longest cached prefix of ``key`` and update LRU state.

        This call may mutate the tree. If ``key`` partially matches a node
        but ends before the node's stored value, that node is split so that the
        returned ``last_node`` corresponds exactly to the matching prefix.

        When ``page_size`` is greater than one the provided ``key`` is truncated
        to the largest multiple of ``page_size`` before matching, mirroring the
        behavior of the original SGLang implementation.
        """
        if self.disable or not key:
            return MatchResult([], self.root_node)

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

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
    def _match_prefix_helper(self, node: TreeNode, key: List[int]) -> Tuple[List[int], TreeNode]:
        node.last_access_time = time.monotonic()
        value: List[int] = []

        while key:
            best_child = None
            best_prefix = 0
            for child in node.children.values():
                prefix_len = self._key_match_fn(child.key, key)
                if prefix_len > best_prefix:
                    best_prefix = prefix_len
                    best_child = child
                    if best_prefix == len(child.key):
                        break

            if not best_child or best_prefix == 0:
                break

            best_child.last_access_time = time.monotonic()
            if best_prefix < len(best_child.key):
                node = self._split_node(best_child, best_prefix)
                value.extend(node.value)
                break
            else:
                value.extend(best_child.value)
                node = best_child
                key = key[best_prefix:]

        return value, node

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        parent = child.parent
        new_node = TreeNode(
            child.key[:split_len],
            child.value[:split_len],
            is_partial=len(child.key[:split_len]) < self.page_size,
        )
        new_node.parent = parent
        new_node.lock_ref = child.lock_ref
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.is_partial = len(child.key) < self.page_size
        child.parent = new_node
        new_node.children[tuple(child.key)] = child
        if parent is not None:
            for k, v in list(parent.children.items()):
                if v is child:
                    del parent.children[k]
                    break
            parent.children[tuple(new_node.key)] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List[int], value: List[int]) -> int:
        node.last_access_time = time.monotonic()
        if not key:
            return 0

        matched = 0
        while key:
            best_child = None
            best_prefix = 0
            for child in node.children.values():
                prefix_len = self._key_match_fn(child.key, key)
                if prefix_len > best_prefix:
                    best_prefix = prefix_len
                    best_child = child
                    if best_prefix == len(child.key):
                        break

            if best_child is None or best_prefix == 0:
                break

            node = best_child
            node.last_access_time = time.monotonic()
            matched += best_prefix
            key = key[best_prefix:]
            value = value[best_prefix:]

            if best_prefix < len(node.key):
                node = self._split_node(node, best_prefix)

            if key and node.is_partial and len(node.key) < self.page_size:
                extend_len = min(self.page_size - len(node.key), len(key))
                old_key = tuple(node.key)
                node.key.extend(key[:extend_len])
                node.value.extend(value[:extend_len])
                node.is_partial = len(node.key) < self.page_size
                if node.parent:
                    del node.parent.children[old_key]
                    node.parent.children[tuple(node.key)] = node
                matched += extend_len
                key = key[extend_len:]
                value = value[extend_len:]

        while key:
            chunk = key[: self.page_size]
            chunk_val = value[: len(chunk)]
            is_partial = len(chunk) < self.page_size
            new_node = TreeNode(list(chunk), list(chunk_val), is_partial=is_partial)
            new_node.parent = node
            node.children[tuple(new_node.key)] = new_node
            self.evictable_size_ += len(chunk_val)
            node = new_node
            matched += len(chunk_val)
            key = key[len(chunk):]
            value = value[len(chunk):]

        return matched

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
    # Key matching helper
    # ------------------------------------------------------------------
    @staticmethod
    def _key_match_seq(k0: Sequence[int], k1: Sequence[int]) -> int:
        i = 0
        for a, b in zip(k0, k1):
            if a != b:
                break
            i += 1
        return i
