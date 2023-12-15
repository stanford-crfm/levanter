import asyncio
import dataclasses
import heapq
import logging
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

import ray
from ray.actor import ActorHandle

from levanter.utils.ray_utils import ExceptionInfo, RayResources, RefBox, current_actor


logger = logging.getLogger(__name__)

T = TypeVar("T")

# We're building a "priority work queue" for Ray. The work queue processes tasks. A task has a priority and a payload.
# The work queue has a function that it applies to the payload, returning a result.
# The work queue processes tasks in priority order. If two tasks have the same priority, we don't care which one gets
# processed first.
# The work queue maintains an elastic pool of actors. When a task arrives, if there is a free actor, it is assigned to
# process the task. If there is no free actor, the task is queued until an actor becomes available. When a task is queued,
# we check to see if we can spin up a new actor. If we can, we do. If we can't, we wait until an actor becomes available.
# When an actor finishes processing a task, it checks to see if there are any queued tasks. If there are, it grabs one
# and processes it. If there aren't, it sleeps until a new task arrives, and spins down after some timeout.

# TODO: a bit more resiliency in case an actor goes down
# TODO: try to schedule a task on a node that has the data it needs
# TODO: purge idle actors if we have too many or they have been idle for too long


@dataclass(order=True, frozen=True)
class _QueueItem:
    priority: float
    batch: ray.ObjectRef = dataclasses.field(compare=False)
    task_id: int
    task_future: asyncio.Future = dataclasses.field(compare=False)


@ray.remote
class PriorityWorkQueue:
    def __init__(
        self,
        process_function: Callable,
        actor_resources: RayResources | dict = RayResources(1),
        min_actors: int = 1,
        max_actors: Optional[int] = None,
    ):
        self.queued_tasks: list = []
        self.process_function = process_function
        if isinstance(actor_resources, dict):
            self.resources = RayResources.from_resource_dict(actor_resources)
        else:
            self.resources = actor_resources
        self._next_task_id = 0
        self._active_tasks = {}  # type: dict[int, asyncio.Future]

        self._next_actor_id = 0
        self.min_actors = min_actors
        self.max_actors = max_actors
        self.workers = {}  # type: dict[int, ActorHandle]
        self.idle_actors = []  # type: list[ActorHandle]
        self.pending_actors = {}  # type: dict[int, ActorHandle]

        if self.min_actors > 0:
            for _ in range(self.min_actors):
                self._spin_up_actor()

    def num_active_actors(self):
        return len(self.workers)

    def num_workers(self):
        return len(self.workers) + len(self.pending_actors)

    def num_idle_workers(self):
        return len(self.idle_actors)

    async def shutdown(self):
        futures = []
        purged = set()
        for actor_set in [self.workers, self.pending_actors]:
            for actor_id, actor in actor_set.items():
                if actor_id not in purged:
                    futures.append(actor.__ray_terminate__.remote())
                    purged.add(actor_id)

        try:
            await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), 5.0)
        except ray.RayTimeoutError:
            logger.exception("Timed out trying to shut down actors. Killing them instead.")
            for actor_set in [self.workers, self.pending_actors]:
                for actor_id, actor in actor_set.items():
                    try:
                        ray.kill(actor)
                    except Exception:
                        logger.exception(f"Failed to kill actor {actor_id}")

    async def submit_task(self, payload: RefBox, priority: float):
        f: asyncio.Future = asyncio.Future()
        task_id = self._next_task_id
        self._next_task_id += 1
        item = _QueueItem(priority, payload.ref, task_id, f)
        heapq.heappush(self.queued_tasks, item)

        self._maybe_assign_tasks()
        return await f

    async def purge_idle_workers(self):
        futures = []
        while len(self.idle_actors) > self.min_actors:
            actor = self.idle_actors.pop()
            actor_id = await actor.get_actor_id.remote()

            self.workers.pop(actor_id, None)
            self.pending_actors.pop(actor_id, None)

            futures.append(actor.__ray_terminate__.remote())

        # ray will throw an exception if we try to get a ray_terminate future for any actor?
        await asyncio.gather(*futures, return_exceptions=True)

    def _maybe_assign_tasks(self):
        while self.queued_tasks and self.idle_actors:
            item = heapq.heappop(self.queued_tasks)
            self._active_tasks[item.task_id] = item.task_future

            actor = self.idle_actors.pop()

            result_ref = actor.apply.remote(item.batch)

            f = asyncio.wrap_future(result_ref.future())
            f.add_done_callback(lambda f: self._task_completed(actor, item.task_id, f.result()))

        if self.queued_tasks:
            self._maybe_start_actor()

    def _maybe_start_actor(self):
        if self._can_allocate_actor():
            self._spin_up_actor()

    # called by the worker actor
    def _task_completed(self, actor: ActorHandle, task_id: int, result):
        task = self._active_tasks.pop(task_id)
        if isinstance(result, ExceptionInfo):
            task.set_exception(result.reraise())
        else:
            task.set_result(result)

        self.idle_actors.append(actor)
        self._maybe_assign_tasks()

    # called by self
    def _can_allocate_actor(self):
        # Implement logic to determine if we can allocate an actor
        # 0. don't try to spin up if we're already spinning up
        if self.pending_actors:
            return False

        current_actors = len(self.workers) + len(self.pending_actors)

        if current_actors < self.min_actors:
            return True

        # 2. we have fewer than max_actors and there are queued tasks
        if not self.queued_tasks:
            return False

        if self.max_actors is not None and current_actors >= self.max_actors:
            return False

        # 3. We have capacity to allocate more actors
        # check ray free resources
        available_resources = ray.available_resources()
        for resource, value in self.resources.to_resource_dict().items():
            if available_resources.get(resource, 0) < value:
                return False

        return True

    def _spin_up_actor(self):
        assert len(self.workers) < self.max_actors
        self_ref = current_actor()

        id = self._next_actor_id
        self._next_actor_id += 1

        actor = _PriorityWorkQueueActor.options(**self.resources.to_kwargs()).remote(self.process_function, id)
        self.pending_actors[id] = actor

        (
            asyncio.wrap_future(actor.get_actor_id.remote().future()).add_done_callback(
                lambda f: self_ref._actor_up.remote(id)
            )
        )
        return actor

    def _actor_up(self, actor_id: int):
        actor = self.pending_actors.pop(actor_id)
        self.workers[actor_id] = actor
        self.idle_actors.append(actor)
        self._maybe_assign_tasks()


@ray.remote
class _PriorityWorkQueueActor:
    def __init__(
        self,
        process_function: Callable,
        actor_id: int,
        idle_timeout: float = 60.0,
    ):
        self.process_function = process_function
        self.actor_id = actor_id
        self.idle_timeout = idle_timeout

    def get_actor_id(self):
        return self.actor_id

    def apply(self, task):
        result = self.process_function(task)
        return result
