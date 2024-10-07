import asyncio
import logging
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import ray


V = TypeVar("V")
R = TypeVar("R")

logger = logging.getLogger(__name__)

# Copilot-Adapted from:
# https://github.com/ray-project/ray/blob/1bab09bf842edee51c3778be4cfb16f8b900d764/python/ray/data/_internal/execution/operators/actor_pool_map_operator.py


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
        self._pending_tasks: List[Tuple[Callable, Any, Optional[ray.ObjectRef], int]] = []
        self._task_futures: Dict[int, asyncio.Future] = {}
        self._next_task_id = 0

        self._scale_up(self._min_size)

    @property
    def num_pending_tasks(self):
        return len(self._pending_tasks)

    def _scale_up(self, num_actors: int):
        for _ in range(num_actors):
            try:
                actor = self._create_actor_fn()
                ready_ref = actor.get_location.remote()
                self._pending_actors[ready_ref] = actor

                async def wait_for_ready(actor, ready_ref):
                    loc = await ready_ref
                    del self._pending_actors[ready_ref]
                    self._push_inner(actor, loc)

                asyncio.ensure_future(wait_for_ready(actor, ready_ref))

            except Exception as e:
                logger.error("Failed to create actor.", exc_info=e)

    def _scale_down(self, num_actors: int):
        for _ in range(num_actors):
            if self._pending_actors:
                actor = self._pending_actors.popitem()[1]
                print(f"Killing pending actor {actor}")
                ray.kill(actor)
            elif self._idle_actors:
                actor = self._idle_actors.pop()
                print(f"Killing idle actor {actor}")
                del self._actor_locations[actor]
                ray.kill(actor)
            else:
                break

    def _adjust_pool_size(self):
        num_pending_tasks = len(self._pending_tasks)
        num_idle_actors = len(self._idle_actors)
        num_busy_actors = len(self._busy_actors)
        num_pending_actors = len(self._pending_actors)

        num_nonworking_actors = num_idle_actors + num_pending_actors

        # TODO: better autoscale logic
        if (
            num_pending_actors == 0
            and num_pending_tasks > 0
            and num_idle_actors == 0
            and num_busy_actors < self._max_size
        ):
            self._scale_up(min(self._max_size - num_busy_actors, num_pending_tasks))
        elif num_pending_tasks == 0 and num_nonworking_actors > self._min_size:
            self._scale_down(num_nonworking_actors - self._min_size)

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
        return actor

    def submit(
        self, fn: Callable[["ray.actor.ActorHandle", V], R], value: V, obj_ref: Optional[ray.ObjectRef] = None
    ) -> asyncio.Future[R]:
        future: asyncio.Future = asyncio.Future()
        actor = self._pick_actor(obj_ref)
        if actor:
            self._assign_task_to_actor(actor, fn, value, future)
        else:
            self._enqueue_pending_task(fn, obj_ref, value, future)
        self._adjust_pool_size()
        return future

    def _enqueue_pending_task(self, fn, obj_ref, value, future):
        task_id = self._next_task_id
        self._next_task_id += 1
        self._task_futures[task_id] = future
        self._pending_tasks.append((fn, value, obj_ref, task_id))

    def _assign_task_to_actor(self, actor, fn, value, future):
        self._idle_actors.remove(actor)
        ray_future = fn(actor, value)
        self._busy_actors[ray_future] = actor
        self._adjust_pool_size()
        asyncio.ensure_future(self._wrap_ray_future(ray_future, future))

    def _maybe_start_pending_task(self, actor):
        if self._pending_tasks:
            fn, value, obj_ref, pending_task_id = self._pending_tasks.pop(0)
            future = self._task_futures.pop(pending_task_id)
            self._assign_task_to_actor(actor, fn, value, future)
            return True
        return False

    async def _wrap_ray_future(self, ray_future, future):
        try:
            result = await ray_future
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            self._on_task_done(ray_future)

    def _on_task_done(self, ray_future):
        actor = self._busy_actors.pop(ray_future)
        self._idle_actors.append(actor)

        started = self._maybe_start_pending_task(actor)
        if not started:
            self._adjust_pool_size()

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

    def pop_idle(self):
        if self._idle_actors:
            return self._idle_actors.pop()
        return None

    def push(self, actor: "ray.actor.ActorHandle"):
        if actor in self._idle_actors or actor in self._busy_actors.values():
            raise ValueError("Actor already belongs to current ActorPool")
        location = ray.get(actor.get_location.remote())

        self._push_inner(actor, location)

    def _push_inner(self, actor, location):
        if actor in self._idle_actors or actor in self._busy_actors.values():
            raise ValueError("Actor already belongs to current ActorPool")
        self._actor_locations[actor] = location
        self._idle_actors.append(actor)
        assigned = self._maybe_start_pending_task(actor)
        if not assigned:
            self._adjust_pool_size()


class PoolWorkerBase(ABC):
    def get_location(self) -> str:
        return ray.get_runtime_context().get_node_id()
