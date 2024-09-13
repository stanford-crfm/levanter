import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator


# Create a ThreadPoolExecutor
_executor = ThreadPoolExecutor(max_workers=10)


def blocking_wait(coro):
    """
    This will only work if there are fewer than 10 levels of nested coroutines...
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        future = _executor.submit(lambda: asyncio.run(coro))
        return future.result()
    else:
        return asyncio.run(coro)


def future_from_value(value):
    future = asyncio.Future()
    future.set_result(value)
    return future


class AsyncIteratorWrapper(Iterator):
    def __init__(self, async_iter):
        self.async_iter = async_iter
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async_task(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._run_async_task(self.async_iter.__anext__())
        except StopAsyncIteration:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()
            raise StopIteration
