import asyncio
import multiprocessing as mp
import queue
import sys
import threading
from multiprocessing import get_context
from queue import Full as QueueFull
from typing import Any, AsyncIterator, Callable, Iterable, Iterator, Literal, Optional, TypeVar, Union

import dill
import tblib


Ex = TypeVar("Ex", covariant=True)

BackgroundMethod = Literal["thread"] | Literal["forkserver"] | Literal["spawn"]


class BackgroundIterable(Iterable[Ex]):
    """
    A wrapper around an iterable that runs the iterable in a background thread/process and fills a queue with the results.

    This allows the iterable to be consumed in a separate thread or process, and for the results to be consumed in the main thread.
    """

    def __init__(
        self,
        producer_fn: Callable[[], Union[Iterator[Ex], AsyncIterator[Ex]]],
        max_capacity: Optional[int] = None,
        method: Optional[BackgroundMethod] = "forkserver",
    ):
        self.max_capacity = max_capacity
        self._producer_fn = producer_fn
        self.method = method

    def __iter__(self) -> "BackgroundIterator[Ex]":
        return BackgroundIterator(self._producer_fn, self.max_capacity, self.method)


class BackgroundIterator(Iterator[Ex]):
    def __init__(
        self,
        producer_fn: Callable[[], Union[Iterator[Ex], AsyncIterator[Ex]]],
        max_capacity: Optional[int],
        method: Optional[BackgroundMethod] = "forkserver",
    ):
        self.max_capacity = max_capacity
        self._producer_fn = producer_fn
        self.method = method
        self._stop_event: Any
        self.q: Union[queue.Queue, mp.Queue]

        match method:
            case None:
                self._stop_event = threading.Event()
                # No background thread/process; consume items on demand
                self.iterator = self._producer_fn()
                if not isinstance(self.iterator, Iterator):
                    # put this here to defer the import for forkserver, since it's slow.
                    from levanter.utils.thread_utils import AsyncIteratorWrapper

                    self.iterator = AsyncIteratorWrapper(self.iterator)
            case "thread":
                self._stop_event = threading.Event()
                self.q = queue.Queue(self.max_capacity or 0)
                self.thread = threading.Thread(target=self._fill_queue_with_batches)
                self.thread.daemon = True
                self.thread.start()
            case meth:
                if meth == "fork":
                    raise ValueError("fork and JAX do not play well together. Use forkserver or spawn")
                elif meth not in ("forkserver", "spawn"):
                    raise ValueError(f"Unknown backgrounding method: {meth}")
                mp.set_forkserver_preload(
                    ["multiprocessing", "dill", "tblib", "sys", "typing", "threading", "time", "asyncio"]
                )
                _ctx = get_context(meth)
                self._stop_event = _ctx.Event()
                self.q = _ctx.Queue(self.max_capacity or 0)
                encoded_fn = dill.dumps(self._producer_fn)
                self.process = _ctx.Process(  # type: ignore
                    target=self._forkserver_worker, args=(encoded_fn, self.q, self._stop_event)
                )
                self.process.daemon = True
                self.process.start()

    def __iter__(self):
        return self

    def __next__(self):
        if self._stop_event.is_set():
            raise StopIteration

        if hasattr(self, "thread") and self.thread is not None:
            # Thread-based queue consumption
            while not self._stop_event.is_set():
                batch = self._do_poll(self.q)
                if batch is _SENTINEL:
                    raise StopIteration
                elif isinstance(batch, _ExceptionWrapper):
                    batch.reraise()
                return batch
        elif hasattr(self, "process") and self.process is not None:
            # Process-based queue consumption
            while not self._stop_event.is_set():
                batch = self._do_poll(self.q)
                if isinstance(batch, _Sentinel):
                    raise StopIteration
                elif isinstance(batch, _ExceptionWrapper):
                    batch.reraise()
                return batch
        else:
            # Consume the iterator directly on demand
            try:
                return next(self.iterator)
            except StopIteration:
                raise
            except StopAsyncIteration:
                raise StopIteration
            except Exception as e:
                raise e
        raise StopIteration

    def __del__(self):
        self.stop()

    def stop(self, wait: bool = True):
        try:
            self._stop_event.set()
        except AttributeError:
            # on mac we get a nonsensical error sometimes?
            pass
        # For threads, join if necessary
        if hasattr(self, "thread") and self.thread is not None and wait and self.thread != threading.current_thread():
            self.thread.join()
        # For processes, join if necessary
        try:
            if hasattr(self, "process") and self.process is not None and wait and self.process.is_alive():
                self.process.join()
        except AttributeError:
            # on mac we get this weird non-sensical error
            # Traceback (most recent call last):
            #   File "/opt/homebrew/Caskroom/miniforge/base/envs/levanter/lib/python3.10/multiprocessing/pool.py", line 271, in __del__
            #   File "/opt/homebrew/Caskroom/miniforge/base/envs/levanter/lib/python3.10/multiprocessing/queues.py", line 371, in put
            # AttributeError: 'NoneType' object has no attribute 'dumps'
            # Exception ignored in: <function BackgroundIterator.__del__ at 0x11c12eef0>
            pass

    def _fill_queue_with_batches(self):
        try:
            iterator = self._producer_fn()
            if isinstance(iterator, Iterator):
                BackgroundIterator._produce_batches_sync(iterator, self.q, self._stop_event, False)
            else:
                asyncio.run(BackgroundIterator._produce_batches_async(iterator, self.q, self._stop_event, False))
        except Exception:
            self.q.put(_ExceptionWrapper(sys.exc_info()))

    @staticmethod
    def _forkserver_worker(encoded_producer_fn, queue, stop_event):
        """
        This is the worker function that runs in a separate process.
        It sets up dill-based serialization and consumes the iterator.
        """

        try:
            producer_fn = dill.loads(encoded_producer_fn)
            iterator = producer_fn()
            if isinstance(iterator, Iterator):
                BackgroundIterator._produce_batches_sync(iterator, queue, stop_event, True)
            else:
                asyncio.run(BackgroundIterator._produce_batches_async(iterator, queue, stop_event, True))
        except Exception:
            queue.put(_ExceptionWrapper(sys.exc_info()))

    @staticmethod
    def _produce_batches_sync(iterator, queue, stop_event, use_multiprocessing):
        try:
            for batch in iterator:
                while not stop_event.is_set():
                    try:
                        BackgroundIterator._do_put(queue, batch, use_multiprocessing)
                        break
                    except QueueFull:
                        pass

                if stop_event.is_set():
                    break

            while not stop_event.is_set():
                try:
                    queue.put(_SENTINEL, block=True, timeout=1)
                    break
                except QueueFull:
                    pass
        except Exception:
            queue.put(_ExceptionWrapper(sys.exc_info()))

    @staticmethod
    async def _produce_batches_async(iterator, queue, stop_event, use_multiprocessing):
        try:
            async for batch in iterator:
                while not stop_event.is_set():
                    try:
                        BackgroundIterator._do_put(queue, batch, use_multiprocessing)
                        break
                    except QueueFull:
                        pass

                if stop_event.is_set():
                    break

            while not stop_event.is_set():
                try:
                    queue.put(_SENTINEL, block=True, timeout=1)
                    break
                except QueueFull:
                    pass
        except Exception:
            queue.put(_ExceptionWrapper(sys.exc_info()))

    @staticmethod
    def _do_put(queue, msg, use_multiprocessing):
        if use_multiprocessing:
            # queue.put(dill.dumps(msg), block=True, timeout=1)
            queue.put(msg, block=True, timeout=1)
        else:
            queue.put(msg, block=True, timeout=1)

    def _do_poll(self, queue):
        if self.method:
            out = queue.get(block=True)
            if isinstance(out, _Sentinel):
                return _SENTINEL
            elif isinstance(out, _ExceptionWrapper):
                return out
            else:
                # return dill.loads(out)
                return out
        else:
            return queue.get(block=True)


class _Sentinel:
    """A sentinel object for marking the end of a stream of data."""

    pass


_SENTINEL = _Sentinel()


class _ExceptionWrapper:
    """Wraps exception and original traceback in object for queue."""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc = exc_info[1]
        self.tb = tblib.Traceback(exc_info[2])

    def reraise(self):
        raise self.exc.with_traceback(self.tb.as_traceback())


_background_process_pool = None


def _get_background_process_pool(ctx):
    global _background_process_pool
    if _background_process_pool is None:
        _background_process_pool = ctx.Pool()
    return _background_process_pool


if __name__ == "__main__":

    async def async_producer():
        for i in range(1, 101):
            yield i

    background_iterable = BackgroundIterable(async_producer, max_capacity=None, method="forkserver")

    iter1 = iter(background_iterable)
    iter2 = iter(background_iterable)

    data1 = [item for item in iter1]
    data2 = [item for item in iter2]

    assert data1 == data2
    assert data1 == list(range(1, 101))
    print(data1)
    print(data2)
    print("done")
    iter1.stop()
    iter2.stop()
    print("stopped")
    iter3 = iter(background_iterable)
    data3 = [item for item in iter3]
    print(data3)
