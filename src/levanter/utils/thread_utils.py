import asyncio
from concurrent.futures import ThreadPoolExecutor


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
