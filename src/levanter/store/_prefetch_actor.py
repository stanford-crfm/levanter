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
    def __init__(self, producer: Callable[[], Iterator[T]], max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.queue: Queue = Queue(maxsize=max_queue_size)  # [T | _Sentinel | _PrefetchException]
        self.producer = _run_producer.remote(self.queue, producer)
        self._stopped = False
        self._finished = False

    def queue_size(self):
        return self.queue.qsize()

    def get_next(self):
        """
        Get the next item from the producer. If the producer raises an exception, it will be reraised here.

        If the producer is done, this will raise StopIteration.
        """
        if self._finished:
            raise StopIteration
        item = self.queue.get()
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


@ray.remote
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
