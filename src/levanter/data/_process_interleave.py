import asyncio
import heapq
from typing import Generic, Optional, Sequence, TypeVar

import ray


G = TypeVar("G")
T = TypeVar("T")


# this is what we want:
# shards.permute().group(G).flatmap_interleaved(f, num_workers)  # produces an iterator over T


# TODO: can we work with this?

# def flatmap_interleaved(f, iterable, *, num_workers, ray_remote_args=None):
#     """Apply f to each element of iterable, returning an interleaved list of results.
#
#     Args:
#         f: A function to apply to each element of iterable. Should return an iterator
#         iterable: An iterable of elements to apply f to.
#         num_workers: The number of workers to use.
#
#     Returns:
#         iterator over the results of applying f to each element of iterable, interleaving the results
#     """
#     iterable = list(enumerate(iterable))
#     # group the elements by worker
#     grouped = [iterable[i::num_workers] for i in range(num_workers)]
#
#     sink = RoundRobinSink.remote(range(len(iterable)))
#
#     results = [_compute_round_robin.options(**(ray_remote_args or {})).remote(f, group, sink) for group in grouped]
#     ray.get(results)
#
#     return sink._buffer.drain()
#
#
# @ray.remote
# def _compute_round_robin(f, groups, sink):
#     serials = [0] * len(groups)
#     emitters = [(group_id, f(group)) for group_id, group in groups]
#     done_emitters = set()
#
#     while len(done_emitters) < len(groups):
#         for idx in range(len(groups)):
#             group_id, emitter = emitters[idx]
#             if group_id in done_emitters:
#                 continue
#             item = next(emitter, None)
#             if item is None:
#                 done_emitters.add(group_id)
#                 emitters[idx] = (group_id, None)
#                 del emitter
#                 sink.group_total_known(group_id, serials[group_id])
#             else:
#                 sink.append_to_group(group_id, serials[group_id], item)
#                 serials[group_id] += 1


@ray.remote
class RoundRobinSink:
    def __init__(self, groups):
        self._buffer = GroupRoundRobinBuffer(groups)

    def append_to_group(self, group, item_serial, item):
        self._buffer.append_to_group(group, item_serial, item)

    def group_total_known(self, group, total):
        self._buffer.group_total_known(group, total)


class GroupRoundRobinBuffer(Generic[G, T]):
    """
    A buffer that holds items from multiple groups and returns them in a round-robin fashion.
    The groups need not have the same number of items. If a group is exhausted, it is removed from the rotation.
    """

    def __init__(self, groups: Sequence[G]):
        self.groups = list(groups)
        self._current_group = 0
        self.buffers: dict[G, list[tuple[int, T]]] = {group: [] for group in groups}
        self._remaining_groups = set(groups)
        self._totals_written: dict[G, int] = {group: 0 for group in groups}
        self._totals_expected: dict[G, Optional[int]] = {group: None for group in groups}

    def append_to_group(self, group: G, item_serial: int, item: T):
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")

        if group not in self._remaining_groups:
            raise ValueError(f"Group {group} already finished")

        heapq.heappush(self.buffers[group], (item_serial, item))

    def group_total_known(self, group: G, total: int):
        if group not in self.groups:
            raise ValueError(f"Group {group} not in {self.groups}")

        if group not in self._remaining_groups:
            raise ValueError(f"Group {group} already finished: {total} vs {self._totals_expected[group]}")

        self._totals_expected[group] = total

        if self._totals_written[group] == total:
            assert len(self.buffers[group]) == 0
            self._remaining_groups.remove(group)

    def is_finished(self):
        return len(self._remaining_groups) == 0

    def pop(self) -> Optional[T]:
        group = self._next_group_to_read_from()
        if group is None:
            return None

        if len(self.buffers[group]) == 0:
            return None

        cur_serial, item = self.buffers[group][0]

        if cur_serial != self._totals_written[group]:
            return None

        heapq.heappop(self.buffers[group])

        self._totals_written[group] += 1

        if self._totals_written[group] == self._totals_expected[group]:
            assert len(self.buffers[group]) == 0
            assert group in self._remaining_groups
            self._remaining_groups.remove(group)

        self._current_group = (self._current_group + 1) % len(self.groups)

        return item

    def drain(self) -> list[T]:
        items = []
        while True:
            item = self.pop()
            if item is None:
                break
            items.append(item)

        return items

    def _next_group_to_read_from(self):
        if len(self._remaining_groups) == 0:
            return None

        while True:
            group = self.groups[self._current_group]
            if group not in self._remaining_groups:
                assert self._totals_written[group] == self._totals_expected[group]
                assert len(self.buffers[group]) == 0
                self._current_group = (self._current_group + 1) % len(self.groups)
            else:
                break
        return group


_SENTINEL = object()


class _BoxedError:
    def __init__(self, exc):
        self.exc = exc

    def __repr__(self):
        return f"BoxedError({self.exc})"

    def __str__(self):
        return f"BoxedError({self.exc})"

    def __eq__(self, other):
        return isinstance(other, _BoxedError) and self.exc == other.exc

    def __hash__(self):
        return hash(self.exc)


def _is_internal_item(item):
    return item is _SENTINEL or isinstance(item, _BoxedError)


class InProgressSequence(Generic[T]):
    def __init__(self):
        self._buffer: list = []
        self._total_added = 0
        self._promises: dict[int, asyncio.Future] = {}
        self._finished_length: Optional[int] = None
        self._finished_promise = asyncio.Future()

    def append(self, item: T):
        if self._finished_length is not None and len(self._buffer) >= self._finished_length:
            raise IndexError("Index out of range")
        self._buffer.append(item)
        self._total_added += 1
        self._fulfill_promise(len(self._buffer) - 1)

    def to_list(self):
        if not self.is_finished():
            raise ValueError("Not finished")
        return list(self._buffer)

    def set_item(self, idx: int, item: T):
        # self._buffer.append(item)
        # return self._fulfill_promises()

        if idx < 0:
            raise IndexError("Negative indices not supported")

        if self._finished_length is not None and idx >= self._finished_length:
            raise IndexError("Index out of range")

        if idx >= len(self._buffer):
            self._buffer.extend([_SENTINEL] * (idx - len(self._buffer) + 1))

        if self._buffer[idx] is _SENTINEL:
            self._total_added += 1

        self._buffer[idx] = item
        self._fulfill_promise(idx)

    def item_exception(self, idx: int, exc: Exception):
        if idx < 0:
            raise IndexError("Negative indices not supported")

        if self._finished_length is not None and idx >= self._finished_length:
            raise IndexError("Index out of range")

        promise = self._promises.pop(idx, None)
        if promise is not None:
            promise.set_exception(exc)

        if idx >= len(self._buffer):
            self._buffer.extend([_SENTINEL] * (idx - len(self._buffer) + 1))

        self._buffer[idx] = _BoxedError(exc)

        self.set_exception(exc)

    def set_finished_length(self, length):
        if self._finished_length is not None:
            raise ValueError("Finished length already set")
        self._finished_length = length
        return self._flush_promises()

    def set_exception(self, exc: Exception):
        if not self._finished_promise.done():
            self._finished_promise.set_exception(exc)
        for promise in self._promises.values():
            promise.set_exception(exc)

        self._promises.clear()

    def is_finished(self):
        return self._finished_length is not None and len(self._buffer) == self._finished_length

    @property
    def finished_promise(self):
        return self._finished_promise

    def final_length(self):
        return self._finished_length

    def current_length(self):
        return len(self._buffer)

    def get_promise(self, idx):
        if idx < 0:
            raise IndexError("Negative indices not supported")

        if self._finished_length is not None and idx >= self._finished_length:
            raise IndexError("Index out of range")

        if self._finished_promise.done() and self._finished_promise.exception():
            return self._finished_promise

        if idx < len(self._buffer):
            promise = asyncio.Future()
            result = self._buffer[idx]
            if isinstance(result, _BoxedError):
                promise.set_exception(result.exc)
                return promise
            elif result is not _SENTINEL:
                promise.set_result(result)
                return promise

        if idx in self._promises:
            return self._promises[idx]

        promise = asyncio.Future()
        self._promises[idx] = promise
        return promise

    def finalize(self):
        if self._finished_length is None:
            self._finished_length = len(self._buffer)
            self._flush_promises()

        assert (
            self._total_added == self._finished_length
        ), f"Finalize called with {self._total_added} != {self._finished_length}"

    async def get(self, idx):
        if idx < len(self._buffer):
            result = self._buffer[idx]
            if isinstance(result, _BoxedError):
                raise result.exc
            elif result is not _SENTINEL:
                return result

        return await self.get_promise(idx)

    def _fulfill_promise(self, idx):
        promise = self._promises.pop(idx, None)
        if promise is not None:
            promise.set_result(self._buffer[idx])

        if self._total_added == self._finished_length:
            self._finished_promise.set_result(None)

    def _flush_promises(self):
        assert self._finished_length is not None

        if self._total_added == self._finished_length:
            self._finished_promise.set_result(None)

        for idx, promise in self._promises.items():
            if idx < self._finished_length:
                if self._buffer[idx] is not _SENTINEL:
                    promise.set_result(self._buffer[idx])
            else:
                promise.set_exception(IndexError("Index out of range"))
