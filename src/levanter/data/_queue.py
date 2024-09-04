import asyncio
import dataclasses
import heapq
import logging as pylogging
import threading
import time
from dataclasses import dataclass
from queue import PriorityQueue
from typing import List, Optional, Protocol, Sequence, TypeVar

import ray
from ray.actor import ActorHandle

from levanter.utils.ray_utils import RefBox

from ._preprocessor import BatchProcessor


logger = pylogging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class PriorityWorkTaskGroupSpec(Protocol):
    name: str

    def build(self) -> "PriorityWorkTaskGroup":
        raise NotImplementedError()


class PriorityWorkTaskGroup(Protocol):
    name: str
    spec: PriorityWorkTaskGroupSpec

    def items(self) -> Sequence["PriorityWorkItem"]:
        raise NotImplementedError()


class PriorityWorkItem(Protocol):
    name: str
    priority: float
    spec: PriorityWorkTaskGroupSpec

    def execute(self) -> tuple[bool, Optional[ray.ObjectRef]]:
        """
        Returns true if the item is finished, false if it should be rescheduled.
        The object ref is used  (1) to block shutting down the actor too early
        and (2) for backpressure.
        """
        raise NotImplementedError()

    # needs to be sortable by priority
    def __lt__(self, other: "PriorityWorkItem"):
        if self.priority == other.priority:
            return self.name < other.name
        else:
            return self.priority < other.priority

    def __le__(self, other: "PriorityWorkItem"):
        if self.priority == other.priority:
            return self.name <= other.name
        else:
            return self.priority <= other.priority


def _mk_queue_aware_process_task(processor: BatchProcessor[T, U], queue: ActorHandle):
    @ray.remote(num_cpus=processor.num_cpus, num_gpus=processor.num_gpus, resources=processor.resources)
    def process_task(desc, batch: List[T]):
        # pylogging.basicConfig(level=DEFAULT_LOG_LEVEL, format=LOG_FORMAT)
        logger.debug(f"Processing batch {desc}")
        queue.task_running.remote()
        try:
            result = processor(batch)
            logger.debug(f"Finished processing batch {desc}")
            return result
        except Exception as e:
            logger.exception(f"Error while processing batch {desc}")
            raise e
        finally:
            pass

    return process_task


@dataclass(order=True, frozen=True)
class _QueueItem:
    priority: float
    desc: str
    batch: ray.ObjectRef = dataclasses.field(compare=False)
    task_id: int
    task_future: asyncio.Future = dataclasses.field(compare=False)


@ray.remote(num_cpus=0)
class _BatchProcessorQueue:  # (Generic[T]): ray doesn't like generics
    """
    A queue of tasks to be processed by a BatchProcessor.

    BatchProcessorQueue spins up tasks to process batches of data.
    It spins up tasks until it reaches the maximum number of tasks that can be run in parallel.
    It then waits for a task to finish before spinning up another one.
    """

    pqueue: PriorityQueue[_QueueItem]
    processor: BatchProcessor
    _next_task_id: int
    ready: bool  # whether or not we can spin up a new task

    @property
    def batch_size(self):
        return self.processor.batch_size

    def __init__(self, batch_processor: BatchProcessor[T, U]):
        self.pqueue = PriorityQueue()
        self.processor = batch_processor
        self._next_task_id = 0
        self.ready = True  # whether we're ready to ask ray to start a new task
        self_ref = ray.runtime_context.get_runtime_context().current_actor
        self._task_processor = _mk_queue_aware_process_task(batch_processor, self_ref)

    # we don't need/want to dereference the batch, so we wrap it in a RefBox
    # one virtue of doing things this way is that we can let Ray try to schedule the compute near the data.
    async def submit(self, priority: float, desc: str, batch: RefBox):
        """Returns a future that is set to the *ObjectRef* of the processed batch. The future is "complete" when the task
        starts, not when it finishes. You then call ray.get on the future's result to get the actual batch."""
        task_id = self._next_task_id
        self._next_task_id += 1
        f: asyncio.Future = asyncio.Future()
        self.pqueue.put(_QueueItem(priority, desc, batch.ref, task_id, f))
        self._maybe_start_task()
        return await f

    def _maybe_start_task(self):
        if self.ready and not self.pqueue.empty():
            self.ready = False
            item = self.pqueue.get()
            batch = item.batch
            try:
                item.task_future.set_result(self._task_processor.remote(item.desc, batch))
            except Exception as e:
                item.task_future.set_exception(e)

    def task_running(self):
        self.ready = True
        self._maybe_start_task()


@ray.remote(num_cpus=0.5, scheduling_strategy="SPREAD")
class WorkQueueDispatcherActor:
    def __init__(self, max_in_flight: Optional[int] = 200):
        pylogging.basicConfig(level=pylogging.INFO, format=LOG_FORMAT)
        self._queue: list[PriorityWorkItem] = []  # heapq
        self._queue_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._current_item: Optional[PriorityWorkItem] = None
        self._max_in_flight = max_in_flight

        self._max_priority: Optional[float] = None
        self._processing_thread = threading.Thread(target=self._loop, daemon=True)
        self._processing_thread.start()

    def set_max_dispatch_priority(self, max_priority: Optional[float]):
        """
        When the sink is full, we will not dispatch items with a priority higher than this.
        """
        with self._queue_lock:
            self._max_priority = max_priority

    def assign_work(self, group: PriorityWorkTaskGroupSpec):
        items = group.build().items()
        with self._queue_lock:
            for item in items:
                heapq.heappush(self._queue, item)

    def is_group_finished(self, group: PriorityWorkTaskGroupSpec):
        with self._queue_lock:
            if any(item.spec == group for item in self._queue):
                return False

            if self._current_item is not None and self._current_item.spec == group:
                return False

            logger.debug(f"Group {group.name} is finished.")

            return True

    def cancel_work_group(self, group: PriorityWorkTaskGroupSpec):
        # kill all the items in the group
        with self._queue_lock:
            self._queue = [item for item in self._queue if item.spec != group]
            heapq.heapify(self._queue)

    def shutdown(self):
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()

            if self._processing_thread.is_alive():
                self._processing_thread.join()

    def _loop(self: "WorkQueueDispatcherActor"):
        should_sleep = False
        backpressure_queue: list[ray.ObjectRef] = []

        def drain_backpressure_to(count):
            nonlocal backpressure_queue
            while len(backpressure_queue) > count:
                finished, remaining = ray.wait(backpressure_queue, num_returns=1, fetch_local=False)
                backpressure_queue = remaining

        while not self._shutdown_event.is_set():
            if should_sleep:
                time.sleep(0.1)

            drain_backpressure_to(self._max_in_flight)

            with self._queue_lock:
                if len(self._queue) == 0:
                    should_sleep = True
                    continue
                else:
                    should_sleep = False

                item = heapq.heappop(self._queue)
                if self._max_priority is not None and item.priority > self._max_priority:
                    logger.debug(f"Item {item.name} has priority {item.priority} which is too high. Rescheduling.")
                    heapq.heappush(self._queue, item)
                    continue
                self._current_item = item

            try:
                item_is_finished, ref = item.execute()
                if ref is not None:
                    backpressure_queue.append(ref)
            except Exception:
                logger.exception(f"Error while processing {item.name}. Killing all associated work.")
                self.cancel_work_group(item.spec)
                continue

            with self._queue_lock:
                self._current_item = None
                if not item_is_finished:
                    heapq.heappush(self._queue, item)

        logger.debug("Shutting down PriorityProcessorActor. Waiting for backpressure to drain.")
        drain_backpressure_to(0)
        logger.debug("Backpressure drained. Shutting down PriorityProcessorActor.")
