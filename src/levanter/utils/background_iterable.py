import queue
import sys
import threading
from typing import Callable, Iterable, Iterator, Optional, TypeVar

import tblib


Ex = TypeVar("Ex", covariant=True)


class BackgroundIterable(Iterable[Ex]):
    """
    A wrapper around an iterable that runs the iterable in a background thread and fills a queue with the results.

    This allows the iterable to be consumed in a separate thread, and for the results to be consumed in the main thread.
    This will only work particularly well if the main thread is doing some kind of IO or other blocking operation,
    like running XLA kernels...
    """

    def __init__(self, producer_fn: Callable[[], Iterator[Ex]], max_capacity: Optional[int] = None):
        self.max_capacity = max_capacity
        self._stop_event = threading.Event()
        self._producer_fn = producer_fn

    def __iter__(self):
        if self._stop_event.is_set():
            raise RuntimeError("Cannot iterate over a stopped BackgroundIterable")

        q = queue.Queue(self.max_capacity)
        thread = threading.Thread(target=self._fill_queue_with_batches, args=(q,))
        thread.daemon = True
        thread.start()

        while not self._stop_event.is_set():
            batch = q.get()
            if batch is _SENTINEL:
                break
            elif isinstance(batch, _ExceptionWrapper):
                batch.reraise()
            yield batch

    def __del__(self):
        self.stop()

    def stop(self):
        self._stop_event.set()

    def _fill_queue_with_batches(self, q):
        try:
            for batch in self._producer_fn():
                # we don't want to block forever because then we can't stop the thread
                while not self._stop_event.is_set():
                    try:
                        q.put(batch, block=True, timeout=1)
                        break
                    except queue.Full:
                        pass

                if self._stop_event.is_set():
                    break

            while not self._stop_event.is_set():
                try:
                    q.put(_SENTINEL, block=True, timeout=1)
                    break
                except queue.Full:
                    # don't hold up the thread if we can't put the sentinel
                    pass
        except Exception:  # flake8: noqa
            q.put(_ExceptionWrapper(sys.exc_info()))


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
