import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator

import ray

from levanter.utils.ray_utils import ExceptionInfo, ser_exc_info


@dataclass
class _PrefetchException:
    info: ExceptionInfo


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


@ray.remote
class PrefetchIteratorActor:
    def __init__(self, producer: Callable[[], Iterator[ray.ObjectRef]], max_queue_size: int = 100):
        self.producer = producer
        self.max_queue_size = max_queue_size
        self.queue: queue.Queue[ray.ObjectRef | _Sentinel | _PrefetchException] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._producer_thread = threading.Thread(target=self._run_producer)
        self._producer_thread.start()

    def queue_size(self):
        return self.queue.qsize()

    def _run_producer(self):
        try:
            producer = self.producer()
            next_item = _SENTINEL
            while not self._stop_event.is_set():
                if next_item is _SENTINEL:
                    try:
                        next_item = next(producer)
                    except StopIteration:
                        break

                    try:
                        self.queue.put(next_item, timeout=1)
                        next_item = _SENTINEL
                    except queue.Full:
                        pass
        except Exception as e:
            self.queue.put(_PrefetchException(ser_exc_info(e)))

        while not self._stop_event.is_set():
            try:
                self.queue.put(_SENTINEL, timeout=1)
                break
            except queue.Full:
                pass

    def get(self):
        """
        Get the next item from the producer. If the producer raises an exception, it will be reraised here.

        If the producer is done, this will raise StopIteration.
        """
        item = self.queue.get()
        if isinstance(item, _PrefetchException):
            item.info.reraise()
        if item is _SENTINEL:
            raise StopIteration
        return ray.get(item)

    def stop(self):
        self._stop_event.set()
        self._producer_thread.join()

    def is_stopped(self):
        return self._stop_event.is_set() and not self._producer_thread.is_alive()
