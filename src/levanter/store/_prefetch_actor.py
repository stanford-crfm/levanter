import asyncio
import logging
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from typing import Callable, Generic, Iterator, List, Optional, TypeVar

import ray

from levanter.utils.ray_utils import ExceptionInfo, ser_exc_info


T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class _PrefetchException:
    info: ExceptionInfo


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class RayPrefetchQueue(Generic[T]):
    def __init__(
        self, producer: Callable[[], Iterator[T]], max_queue_size: int = 100, producer_options: dict | None = None
    ):
        self.max_queue_size = max_queue_size
        if producer_options is None:
            producer_options = {}
        self.queue_actor = _QueueActor.remote(max_queue_size)  # type: ignore
        self.producer_task = _run_producer.options(**producer_options).remote(self.queue_actor, producer)
        self._stopped = False
        self._finished = False

    def queue_size(self):
        return ray.get(self.queue_actor.qsize.remote())

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
        # time_in = time.time()
        item = ray.get(self.queue_actor.get_next.remote(timeout))
        # time_out = time.time()
        # if time_out - time_in > 0.1:
        #     current_name = ray.get_runtime_context().get_actor_name()
        #     print(f"{current_name} :: Queue get took {time_out - time_in} seconds :: {self.queue_size()}")
        #     logger.info(f"{current_name} :: Queue get took {time_out - time_in} seconds :: {self.queue_size()}")
        if isinstance(item, _PrefetchException):
            item.info.reraise()
        if isinstance(item, _Sentinel):
            self._finished = True
            raise StopIteration
        return item

    def stop(self):
        ray.cancel(self.producer_task)
        ray.get(self.queue_actor.stop.remote())
        self._stopped = True

    def is_stopped(self):
        return self._stopped

    def drain_available(self, max_size: int) -> List[T]:
        return ray.get(self.queue_actor.drain_available.remote(max_size))


@ray.remote
class _QueueActor:
    def __init__(self, max_queue_size: int):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._stopped = False
        self._finished = False

    async def put(self, item):
        await self.queue.put(item)

    async def get_next(self, timeout: Optional[float] = None):
        try:
            if timeout is not None:
                item = await asyncio.wait_for(self.queue.get(), timeout)
            else:
                item = await self.queue.get()
            if isinstance(item, _Sentinel):
                self._finished = True
            return item
        except asyncio.TimeoutError:
            raise QueueEmpty

    async def drain_available(self, max_size: int) -> List[T]:
        items: list[T] = []
        while len(items) < max_size:
            try:
                item = self.queue.get_nowait()
                if isinstance(item, _Sentinel):
                    self._finished = True
                    break
                if isinstance(item, _PrefetchException):
                    item.info.reraise()
                items.append(item)
            except asyncio.QueueEmpty:
                break
        return items

    async def qsize(self):
        return self.queue.qsize()

    async def stop(self):
        self._stopped = True


@ray.remote
def _run_producer(queue_actor, producer_fn: Callable[[], Iterator[T]]):
    async def _run_producer(queue_actor, producer_fn):
        previous_put = None
        try:
            producer = producer_fn()
            del producer_fn

            while True:
                next_item = next(producer)
                if previous_put is not None:
                    await previous_put
                previous_put = queue_actor.put.remote(next_item)
        except StopIteration:
            if previous_put is not None:
                await previous_put
            await queue_actor.put.remote(_SENTINEL)
        except Exception as e:
            if previous_put is not None:
                await previous_put
            await queue_actor.put.remote(_PrefetchException(ser_exc_info(e)))

    asyncio.run(_run_producer(queue_actor, producer_fn))
