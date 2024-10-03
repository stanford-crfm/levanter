import asyncio

import pytest
import ray

from levanter.utils.actor_pool import AutoScalingActorPool, PoolWorkerBase
from levanter.utils.py_utils import logical_cpu_core_count


@ray.remote
class TestActor(PoolWorkerBase):
    def __init__(self):
        self.node_id = ray.get_runtime_context().get_node_id()

    def get_node_id(self):
        return self.node_id

    def double(self, v):
        print(v)
        return 2 * v


@ray.remote
class BlockerActor(PoolWorkerBase):
    def __init__(self):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.unblocked = False
        self.unblock_event = asyncio.Event()

    def get_node_id(self):
        return self.node_id

    async def block(self):
        if not self.unblocked:
            await self.unblock_event.wait()

    async def unblock(self):
        self.unblocked = True
        self.unblock_event.set()


@ray.remote
class BlockingTestActor(PoolWorkerBase):
    def __init__(self, blocker):
        self.node_id = ray.get_runtime_context().get_node_id()
        self.blocker = blocker

    def get_node_id(self):
        return self.node_id

    def double(self, v, bypass_blocker=False):
        print(v)
        if not bypass_blocker:
            ray.get(self.blocker.block.remote())
        print(f"Unblocked {v}")
        return 2 * v


# Helper function to create a TestActor
def create_test_actor():
    return TestActor.remote()


def create_test_actor_blocker(blocker_handle):
    return BlockingTestActor.remote(blocker_handle)


def setup_module(module):
    ray.init(
        "local", num_cpus=max(2 * logical_cpu_core_count(), 8), ignore_reinit_error=True
    )  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


@pytest.mark.asyncio
async def test_basic_submit():
    pool = AutoScalingActorPool(create_test_actor, min_size=1, max_size=4)
    results = [pool.submit(lambda a, v: a.double.remote(v), i) for i in range(4)]
    results = [await r for r in results]

    assert results == [0, 2, 4, 6]


@pytest.mark.asyncio
async def test_basic_submit_no_idle():
    pool = AutoScalingActorPool(create_test_actor, min_size=0, max_size=4)
    results = [pool.submit(lambda a, v: a.double.remote(v), i) for i in range(4)]
    results = [await r for r in results]

    assert results == [0, 2, 4, 6]


@pytest.mark.asyncio
async def test_basic_functionality():
    pool = AutoScalingActorPool(create_test_actor, min_size=1, max_size=4)
    results = list(await pool.map(lambda a, v: a.double.remote(v), [1, 2, 3, 4]))
    assert results == [2, 4, 6, 8]


@pytest.mark.asyncio
async def test_scaling_up():
    blocker = BlockerActor.remote()
    pool = AutoScalingActorPool(lambda: create_test_actor_blocker(blocker), min_size=1, max_size=4)
    f1 = pool.submit(lambda a, v: a.double.remote(v), 1)
    f2 = pool.submit(lambda a, v: a.double.remote(v), 2)
    f3 = pool.submit(lambda a, v: a.double.remote(v, True), 3)
    f4 = pool.submit(lambda a, v: a.double.remote(v, True), 4)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(f2), timeout=0.1)

    assert (await asyncio.gather(f3, f4)) == [6, 8]

    await blocker.unblock.remote()
    assert (await asyncio.gather(f1, f2)) == [2, 4]


@pytest.mark.asyncio
async def test_scaling_down():
    pool = AutoScalingActorPool(create_test_actor, min_size=1, max_size=4)
    await pool.submit(lambda a, v: a.double.remote(v), 1)
    await pool.submit(lambda a, v: a.double.remote(v), 2)
    await pool.submit(lambda a, v: a.double.remote(v), 3)
    await pool.submit(lambda a, v: a.double.remote(v), 4)
    results = await asyncio.gather(
        pool.submit(lambda a, v: a.double.remote(v), 1),
        pool.submit(lambda a, v: a.double.remote(v), 2),
        pool.submit(lambda a, v: a.double.remote(v), 3),
        pool.submit(lambda a, v: a.double.remote(v), 4),
    )
    assert results == [2, 4, 6, 8]
    assert len(pool._idle_actors) == 1
    assert len(pool._busy_actors) == 0


@pytest.mark.asyncio
async def test_push_pop_idle():
    pool = AutoScalingActorPool(create_test_actor, min_size=1, max_size=4)
    actor = pool.pop_idle()
    assert actor is not None
    pool.push(actor)
    assert len(pool._idle_actors) == 1


@pytest.mark.asyncio
async def test_has_free():
    block_actor = BlockerActor.remote()
    pool = AutoScalingActorPool(lambda: create_test_actor_blocker(block_actor), min_size=1, max_size=1)
    assert pool.has_free()
    f = pool.submit(lambda a, v: a.double.remote(v), 1)
    assert not pool.has_free()
    await block_actor.unblock.remote()
    await f
    assert pool.has_free()


@pytest.mark.asyncio
async def test_submit_with_no_idle_actors():
    blocker = BlockerActor.remote()
    pool = AutoScalingActorPool(lambda: create_test_actor_blocker(blocker), min_size=1, max_size=4)
    futs = [pool.submit(lambda a, v: a.double.remote(v), i) for i in range(4)]
    f5 = pool.submit(lambda a, v: a.double.remote(v), 5)
    assert len(pool._pending_tasks) == 1
    await blocker.unblock.remote()
    await asyncio.gather(*futs)
    assert len(pool._pending_tasks) == 0
    assert (await f5) == 10
