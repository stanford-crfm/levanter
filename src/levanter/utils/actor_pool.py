import asyncio
import logging
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, TypeVar

import ray


V = TypeVar("V")
R = TypeVar("R")

logger = logging.getLogger(__name__)

# Copilot-Adapted from:
# https://github.com/ray-project/ray/blob/1bab09bf842edee51c3778be4cfb16f8b900d764/python/ray/data/_internal/execution/operators/actor_pool_map_operator.py


def _wrap_ray_future(ray_future):
    # work around https://github.com/ray-project/ray/issues/45895#issuecomment-2165164129
    return asyncio.wrap_future(ray_future.future())


class AutoScalingActorPool:
    """Utility class to operate on a dynamically scaling pool of actors."""

    def __init__(
        self,
        create_actor_fn: Callable[[], "ray.actor.ActorHandle"],
        min_size: int = 1,
        max_size: int = 10,
    ):
        if max_size < min_size:
            raise ValueError("max_size must be greater than or equal to min_size.")
        self._create_actor_fn = create_actor_fn
        self._min_size = min_size
        self._max_size = max_size

        self._idle_actors: List[ray.actor.ActorHandle] = []
        self._busy_actors: Dict[ray.ObjectRef, ray.actor.ActorHandle] = {}
        self._pending_actors: Dict[ray.ObjectRef, ray.actor.ActorHandle] = {}

        self._actor_locations: Dict[ray.actor.ActorHandle, str] = {}
        self._tasks_waiting_for_actor: list[asyncio.Future] = []
        self._next_task_id = 0
        self._scale_down_task: Optional[asyncio.Task] = None

        self._scale_up(self._min_size)

    @property
    def num_pending_tasks(self):
        return len(self._tasks_waiting_for_actor)

    def resize_pool(self, *, min_size: Optional[int] = None, max_size: Optional[int] = None):
        old_min_size = self._min_size
        if min_size is not None:
            self._min_size = min_size
        old_max_size = self._max_size
        if max_size is not None:
            self._max_size = max_size

        if old_min_size != self._min_size or old_max_size != self._max_size:
            logger.info(f"Resizing pool to min_size: {self._min_size}, max_size: {self._max_size}")

        self._adjust_pool_size()

    def get_max_size(self):
        return self._max_size

    def get_min_size(self):
        return self._min_size

    def _scale_up(self, num_actors: int):
        if self._scale_down_task and not self._scale_down_task.done():
            self._scale_down_task.cancel()

        for _ in range(num_actors):
            try:
                actor = self._create_actor_fn()
                ready_ref = actor.get_location.remote()
                self._pending_actors[ready_ref] = actor

                async def wait_for_ready(actor, ready_ref):
                    loc = await _wrap_ray_future(ready_ref)
                    # pending -> floating
                    if ready_ref not in self._pending_actors:
                        logger.info("Actor was cancelled before it was ready.")
                        return
                    del self._pending_actors[ready_ref]
                    self._assert_is_floating(actor)
                    self._actor_locations[actor] = loc
                    self._maybe_start_pending_task(actor)  # floating -> {idle, busy}

                asyncio.ensure_future(wait_for_ready(actor, ready_ref))

            except Exception as e:
                logger.error("Failed to create actor.", exc_info=e)

    def _scale_down(self, target_num_actors: int):
        while len(self._idle_actors) + len(self._pending_actors) > target_num_actors:
            if self._pending_actors:
                actor = self._pending_actors.popitem()[1]
                # let it die through gc
                # ray.kill(actor)
            elif self._idle_actors:
                actor = self._idle_actors.pop()
                del self._actor_locations[actor]
                # let it die through gc
                # ray.kill(actor)
            else:
                break

    def _adjust_pool_size(self):
        num_pending_tasks = self.num_pending_tasks
        num_idle_actors = len(self._idle_actors)
        num_busy_actors = len(self._busy_actors)
        num_pending_actors = len(self._pending_actors)

        num_nonworking_actors = num_idle_actors + num_pending_actors
        total_actors = num_nonworking_actors + num_busy_actors

        # TODO: better autoscale logic
        if (
            num_pending_actors == 0
            and num_pending_tasks > 0
            and num_idle_actors == 0
            and total_actors < self._max_size
        ):
            logger.info(
                f"Scaling up due to {num_pending_tasks} pending tasks. Current pool size: {total_actors}. Max size:"
                f" {self._max_size}"
            )
            self._scale_up(min(self._max_size - num_busy_actors, num_pending_tasks))

        # Schedule scale down if idle
        elif num_pending_tasks == 0 and num_nonworking_actors > self._min_size:
            if self._scale_down_task is None:
                self._scale_down_task = asyncio.create_task(self._schedule_scale_down())

    async def _schedule_scale_down(self):
        try:
            await asyncio.sleep(10)
            if self.num_pending_tasks == 0:
                logger.info("Scaling down due to no pending tasks.")
                self._scale_down(self._min_size)
                self._scale_down_task = None
        except asyncio.CancelledError:
            logger.debug("Scale down task was cancelled due to new activity.")

    def _get_object_location(self, obj_ref: ray.ObjectRef) -> Optional[str]:
        """Get the location of the given object reference."""
        try:
            locs = ray.experimental.get_object_locations([obj_ref])
            nodes = locs[obj_ref]["node_ids"]
            if nodes:
                return nodes[0]
        except Exception as e:
            logger.error(f"Failed to get object location: {e}")
        return None

    def _pick_actor(self, obj_ref: Optional[ray.ObjectRef] = None) -> Optional[ray.actor.ActorHandle]:
        """Pick an actor based on locality and busyness."""
        # idle -> floating
        if not self._idle_actors:
            return None

        if obj_ref:
            preferred_loc = self._get_object_location(obj_ref)
        else:
            preferred_loc = None

        def penalty_key(actor):
            """Returns the key that should be minimized for the best actor."""
            requires_remote_fetch = self._actor_locations[actor] != preferred_loc
            return requires_remote_fetch

        actor = min(self._idle_actors, key=penalty_key)
        actor = self._idle_actors.pop(self._idle_actors.index(actor))
        return actor

    def submit(self, fn: Callable[["ray.actor.ActorHandle", V], R], value: V, obj_ref: Optional[ray.ObjectRef] = None):
        actor = self._pick_actor(obj_ref)
        if actor:
            return self._assign_task_to_actor(actor, fn, value)
        else:
            actor_future: asyncio.Future = asyncio.Future()
            self._tasks_waiting_for_actor.append(actor_future)
            f = asyncio.ensure_future(self._enqueue_pending_task(fn, obj_ref, value, actor_future))
            self._adjust_pool_size()
            return f

    def _assign_task_to_actor(self, actor, fn, value):
        # floating -> busy
        ray_future = fn(actor, value)
        self._busy_actors[ray_future] = actor
        if self._scale_down_task and not self._scale_down_task.done():
            self._scale_down_task.cancel()
        self._adjust_pool_size()

        return asyncio.ensure_future(self._set_up_actor_return_on_finished(ray_future))

    async def _enqueue_pending_task(self, fn, obj_ref, value, actor_future):
        actor = await actor_future
        return await self._assign_task_to_actor(actor, fn, value)

    def _assert_is_floating(self, actor):
        assert actor not in self._idle_actors
        assert actor not in self._busy_actors
        assert actor not in self._pending_actors

    def _maybe_start_pending_task(self, actor):
        self._assert_is_floating(actor)
        if self._tasks_waiting_for_actor:
            # floating -> busy (inside the _enqueue_pending_task coroutine)
            actor_future = self._tasks_waiting_for_actor.pop(0)
            actor_future.set_result(actor)
            assigned = True
        else:
            # floating -> idle
            self._idle_actors.append(actor)
            self._adjust_pool_size()
            assigned = False
        return assigned

    async def _set_up_actor_return_on_finished(self, ray_future):
        future = _wrap_ray_future(ray_future)
        await asyncio.wait([future])
        self._on_task_done(ray_future)
        return await future

    def _on_task_done(self, ray_future):
        actor = self._busy_actors.pop(ray_future)
        self._maybe_start_pending_task(actor)

    async def map(
        self,
        fn: Callable[["ray.actor.ActorHandle", V], Any],
        values: List[V],
        obj_refs: Optional[List[Optional[ray.ObjectRef]]] = None,
    ) -> List[Any]:
        if obj_refs is None:
            obj_refs = [None] * len(values)

        tasks = [self.submit(fn, v, obj_ref) for v, obj_ref in zip(values, obj_refs)]
        return await asyncio.gather(*tasks)

    def has_free(self):
        return bool(self._idle_actors)

    def has_free_or_pending_actors(self):
        return bool(self._idle_actors) or bool(self._pending_actors)

    def pop_idle(self):
        if self._idle_actors:
            return self._idle_actors.pop()
        return None

    def push(self, actor: "ray.actor.ActorHandle"):
        location = ray.get(actor.get_location.remote())
        self._actor_locations[actor] = location
        self._maybe_start_pending_task(actor)

    def __del__(self):
        if self._scale_down_task and not self._scale_down_task.done():
            self._scale_down_task.cancel()
        # just let ray kill the actors naturally


class PoolWorkerBase(ABC):
    def get_location(self) -> str:
        return ray.get_runtime_context().get_node_id()
