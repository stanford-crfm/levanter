import asyncio

import pytest

from levanter.utils.background_iterable import BackgroundIterable


def test_reentrancy():
    test_data = list(range(1, 101))
    background_iterable = BackgroundIterable(lambda: iter(test_data), max_capacity=10)

    iter1 = iter(background_iterable)
    iter2 = iter(background_iterable)

    data1 = list(iter1)
    data2 = list(iter2)

    assert data1 == data2
    assert data1 == test_data


def test_empty_iteration():
    # Create a BackgroundIterable instance with an empty producer function
    background_iterable = BackgroundIterable(lambda: iter([]), max_capacity=10)

    # Convert the iterator to a list for comparison
    data = list(background_iterable)

    # Assert that the produced data is empty
    assert data == []


def test_exception_handling():
    # Create a producer function that raises an exception
    def producer_with_exception():
        raise ValueError("Something went wrong!")

    # Create a BackgroundIterable instance with the producer function that raises an exception
    background_iterable = BackgroundIterable(producer_with_exception, max_capacity=10)

    # Iterate over the BackgroundIterable and handle the raised exception
    with pytest.raises(ValueError):
        for _ in background_iterable:
            pass


def test_stop_event():
    def ongoing_process():
        while True:
            for item in range(1, 101):
                yield item

    background_iterable = BackgroundIterable(ongoing_process, max_capacity=10)

    iter1 = iter(background_iterable)

    for _ in range(5):
        next(iter1)

    iter1.stop()

    # Try to get another item from the iterator (should raise StopIteration)
    # there's a bit of a race so we give it 2 tries, which is enough for the test
    with pytest.raises(StopIteration):
        next(iter1)
        next(iter1)


@pytest.mark.asyncio
async def test_async_reentrancy():
    async def async_producer():
        for i in range(1, 101):
            yield i
            await asyncio.sleep(0.01)

    background_iterable = BackgroundIterable(async_producer, max_capacity=10)

    iter1 = iter(background_iterable)
    iter2 = iter(background_iterable)

    data1 = [item for item in iter1]
    data2 = [item for item in iter2]

    assert data1 == data2
    assert data1 == list(range(1, 101))


@pytest.mark.asyncio
async def test_async_empty_iteration():
    async def async_producer():
        if False:
            yield

    background_iterable = BackgroundIterable(async_producer, max_capacity=10)

    data = list(background_iterable)

    assert data == []


@pytest.mark.asyncio
async def test_async_exception_handling():
    async def async_producer_with_exception():
        raise ValueError("Something went wrong!")
        yield 0  # have to make sure it's an async coroutine

    background_iterable = BackgroundIterable(async_producer_with_exception, max_capacity=10)

    with pytest.raises(ValueError):
        for _ in background_iterable:
            pass


@pytest.mark.asyncio
async def test_async_stop_event():
    async def ongoing_async_process():
        while True:
            for item in range(1, 101):
                yield item

    background_iterable = BackgroundIterable(ongoing_async_process, max_capacity=10)

    iter1 = iter(background_iterable)

    for _ in range(5):
        next(iter1)

    iter1.stop()

    with pytest.raises(StopIteration):
        await next(iter1)
        await next(iter1)
