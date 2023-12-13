import asyncio
import os
import time

import pytest
import ray

from levanter.data._priority_work_queue import PriorityWorkQueue
from levanter.utils.ray_utils import RefBox


@pytest.fixture(scope="module")
def ray_local_cluster():
    # Set up a local Ray cluster
    RUNNING_IN_CI = "CI" in os.environ
    ray.init("local", num_cpus=4, num_gpus=0, include_dashboard=not RUNNING_IN_CI)
    yield
    # Tear down the Ray cluster
    ray.shutdown()


def test_initialization(ray_local_cluster):
    # Test the initialization of the work queue with min and max actors
    process_function = lambda x: x  # noqa: E731
    min_actors = 2
    max_actors = 5
    work_queue = PriorityWorkQueue.remote(process_function, {}, min_actors, max_actors)

    assert ray.get(work_queue.num_workers.remote()) == min_actors


def test_task_submission_and_processing(ray_local_cluster):
    process_function = lambda x: x * 2  # noqa: E731
    work_queue = PriorityWorkQueue.remote(process_function, {}, 1, 3)

    # Submit a task and check if it gets processed correctly
    payload = ray.put(10)
    result = ray.get(work_queue.submit_task.remote(RefBox(payload), 1.0))

    assert result == 20  # Expecting the processed result
    # Add more assertions as needed


def test_dynamic_actor_allocation(ray_local_cluster):
    @ray.remote(num_cpus=0)
    class Blocker:
        def __init__(self, count):
            self.future = asyncio.Future()
            self.count_met = asyncio.Future()
            self.count = count

        def get_count(self):
            return self.count

        async def await_count_met(self):
            await self.count_met

        async def block(self):
            self.count -= 1
            if self.count == 0:
                self.count_met.set_result(None)
            await self.future

        def unblock(self):
            self.future.set_result(None)

    blocker_to_wait_on_test = Blocker.remote(3)

    def process_function(x):
        ray.get(blocker_to_wait_on_test.block.remote())
        return x * 2

    work_queue = PriorityWorkQueue.remote(process_function, {}, 1, 3)

    # Submit multiple tasks and check actor allocation
    futures = []
    for i in range(5):
        payload = ray.put(i)
        futures.append(work_queue.submit_task.remote(RefBox(payload), i))

    ray.get(blocker_to_wait_on_test.await_count_met.remote())

    for tries in range(5):
        if ray.get(work_queue.num_workers.remote()) == 3:
            break
        else:
            time.sleep(0.1)
    else:
        pytest.fail("Expected 3 actors to be allocated")

    assert ray.get(work_queue.num_idle_workers.remote()) == 0  # All actors should be busy

    ray.get(blocker_to_wait_on_test.unblock.remote())

    results = ray.get(futures)

    assert results == [0, 2, 4, 6, 8]  # Expecting the processed results

    assert ray.get(work_queue.num_workers.remote()) == 3  # There should be 3 actors
    assert ray.get(work_queue.num_idle_workers.remote()) == 3  # All actors should be idle

    ray.get(work_queue.purge_idle_workers.remote())

    assert ray.get(work_queue.num_workers.remote()) == 1  # There should be 1 actor
    assert ray.get(work_queue.num_idle_workers.remote()) == 1  # All actors should be idle

    ray.get(work_queue.shutdown.remote())
