import time
from typing import Iterator

import pytest
import ray

from levanter.store._prefetch_actor import RayPrefetchQueue


def _sleep_until(condition, timeout=5, message="Condition not met within timeout"):
    start = time.time()
    while not condition():
        if time.time() - start > timeout:
            pytest.fail(message)
        time.sleep(0.1)


@pytest.fixture(scope="module", autouse=True)
def ray_init_and_shutdown():
    ray.init("local", ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.ray
def test_initialization_and_basic_functionality():
    def simple_producer():
        for i in range(10):
            yield i

    actor = RayPrefetchQueue(simple_producer)
    results = [actor.get_next() for _ in range(10)]
    assert results == list(range(10))


@pytest.mark.ray
def test_queue_size_limit():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(100):
            yield i

    actor = RayPrefetchQueue(simple_producer, max_queue_size=10)
    # Allow some time for the queue to fill up
    _sleep_until(lambda: actor.queue_size() == 10)

    # get a few items to make some space
    [actor.get_next() for _ in range(5)]
    _sleep_until(lambda: actor.queue_size() == 10, message="Queue size did not reach 10")


@pytest.mark.ray
def test_stop_functionality():
    def simple_producer():
        for i in range(10000):
            yield i

    actor = RayPrefetchQueue(simple_producer)
    actor.stop()

    _sleep_until(lambda: actor.is_stopped(), message="Actor did not stop")


@pytest.mark.ray
def test_exception_handling():
    def faulty_producer():
        for i in range(5):
            yield i
        raise ValueError("Test exception")

    actor = RayPrefetchQueue(faulty_producer)
    results = []
    try:
        for _ in range(10):
            results.append(actor.get_next())
    except ValueError as e:
        assert "Test exception" in str(e)  # Ray puts a lot of crap in the exception message
    assert results == list(range(5))


@pytest.mark.ray
def test_empty_producer():
    def empty_producer() -> Iterator[ray.ObjectRef]:
        if False:
            yield

    actor = RayPrefetchQueue(empty_producer)
    with pytest.raises(StopIteration):
        actor.get_next()


@pytest.mark.ray
def test_multiple_consumers():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(20):
            yield i

    actor = RayPrefetchQueue(simple_producer)
    results = [actor.get_next() for _ in range(10)]
    results += [actor.get_next() for _ in range(10)]
    assert results == list(range(20))


@pytest.mark.ray
def test_producer_completion():
    def simple_producer():
        for i in range(10):
            yield i

    actor = RayPrefetchQueue(simple_producer)
    results = []
    try:
        while True:
            results.append(actor.get_next())
    except StopIteration:
        pass
    assert results == list(range(10))


@pytest.mark.ray
def test_drain_queue():
    def simple_producer():
        for i in range(10):
            yield i

    actor = RayPrefetchQueue(simple_producer)

    all_results = []

    for tot in range(0, 5):
        out = actor.drain_available(tot)
        assert len(out) <= tot
        all_results.extend(out)

    while len(all_results) < 10:
        all_results.append(actor.get_next())

    assert all_results == list(range(10))
