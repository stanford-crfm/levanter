from dataclasses import dataclass
from typing import Callable, Generic, Iterator, TypeVar

import ray
from ray.util.queue import Queue

from levanter.utils.ray_utils import ExceptionInfo, ser_exc_info


T = TypeVar("T")


@dataclass
class _PrefetchException:
    info: ExceptionInfo


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class RayPrefetchQueue(Generic[T]):
    def __init__(self, producer: Callable[[], Iterator[T]], max_queue_size: int = 100, name: str | None = None):
        self.max_queue_size = max_queue_size
        if name is not None:
            actor_options = {"name": f"{name}::queue"}
            producer_options = {"name": f"{name}::producer"}
        else:
            actor_options = {}
            producer_options = {}
        self.queue: Queue = Queue(
            maxsize=max_queue_size, actor_options=actor_options
        )  # [T | _Sentinel | _PrefetchException]
        self.producer = _run_producer.options(**producer_options).remote(self.queue, producer)
        self._stopped = False
        self._finished = False

    def queue_size(self):
        return self.queue.qsize()

    def __next__(self):
        return self.get_next()

    def __iter__(self):
        return self

    def get_next(self, timeout: float | None = None) -> T:
        """
        Get the next item from the producer. If the producer raises an exception, it will be reraised here.

        If the producer is done, this will raise StopIteration.

        Args:
            timeout (float|None): Timeout in seconds for getting the next item. If None, will block indefinitely.

        Raises:
            Empty: If the queue is empty and the timeout is reached.
        """
        if self._finished:
            raise StopIteration
        item = self.queue.get(timeout=timeout)
        if isinstance(item, _PrefetchException):
            item.info.reraise()
        if isinstance(item, _Sentinel):
            self._finished = True
            raise StopIteration
        return item

    def stop(self):
        ray.cancel(self.producer)
        self.queue.shutdown()
        self._stopped = True

    def is_stopped(self):
        return self._stopped


@ray.remote(scheduling_strategy="SPREAD")
def _run_producer(queue: Queue, producer_fn: Callable[[], Iterator[T]]):
    try:
        producer = producer_fn()
        del producer_fn

        while True:
            next_item = next(producer)
            queue.put(next_item)
    except StopIteration:
        queue.put(_SENTINEL)
    except Exception as e:
        queue.put(_PrefetchException(ser_exc_info(e)))
        raise
