import time
from typing import Iterator

import pytest
import ray

from levanter.store._prefetch_actor import PrefetchIteratorActor


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


def test_initialization_and_basic_functionality():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(10):
            yield ray.put(i)

    actor = PrefetchIteratorActor.remote(simple_producer)
    results = ray.get([actor.get.remote() for _ in range(10)])
    assert results == list(range(10))


def test_queue_size_limit():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(100):
            yield ray.put(i)

    actor = PrefetchIteratorActor.remote(simple_producer, max_queue_size=10)
    # Allow some time for the queue to fill up
    _sleep_until(lambda: ray.get(actor.queue_size.remote()) == 10)
    assert ray.get(actor.queue_size.remote()) == 10

    # get a few items to make some space
    ray.get([actor.get.remote() for _ in range(5)])
    _sleep_until(lambda: ray.get(actor.queue_size.remote()) == 10, message="Queue size did not reach 10")
    assert ray.get(actor.queue_size.remote()) == 10


def test_stop_functionality():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(10):
            yield ray.put(i)

    actor = PrefetchIteratorActor.remote(simple_producer)
    ray.get(actor.stop.remote())

    _sleep_until(lambda: ray.get(actor.is_stopped.remote()), message="Actor did not stop")


def test_exception_handling():
    def faulty_producer() -> Iterator[ray.ObjectRef]:
        for i in range(5):
            yield ray.put(i)
        raise ValueError("Test exception")

    actor = PrefetchIteratorActor.remote(faulty_producer)
    results = []
    try:
        for _ in range(10):
            results.append(ray.get(actor.get.remote()))
    except ValueError as e:
        assert "Test exception" in str(e)  # Ray puts a lot of crap in the exception message
    assert results == list(range(5))


def test_empty_producer():
    def empty_producer() -> Iterator[ray.ObjectRef]:
        if False:
            yield

    actor = PrefetchIteratorActor.remote(empty_producer)
    with pytest.raises(StopIteration):
        ray.get(actor.get.remote())


def test_multiple_consumers():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(20):
            yield ray.put(i)

    actor = PrefetchIteratorActor.remote(simple_producer)
    results = ray.get([actor.get.remote() for _ in range(10)])
    results += ray.get([actor.get.remote() for _ in range(10)])
    assert results == list(range(20))


def test_producer_completion():
    def simple_producer() -> Iterator[ray.ObjectRef]:
        for i in range(10):
            yield ray.put(i)

    actor = PrefetchIteratorActor.remote(simple_producer)
    results = []
    try:
        while True:
            results.append(ray.get(actor.get.remote()))
    except StopIteration:
        pass
    assert results == list(range(10))
