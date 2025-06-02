from __future__ import annotations

from ray.actor import ActorHandle

from levanter.utils.ray_utils import current_actor_handle

"""Flexible multi‑slice TPU orchestration on Ray (with CPU‑only simulation).

This module supports two modes:
1. Real TPU clusters using Ray‑TPU resources via RayTpuManager.
2. CPU‑only simulation via `--simulate` and `--sim_slices`.

Run a demo:
  python -m levanter.infra.tpu_flexible_cluster --simulate --min 2 --max 3
"""
import argparse
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import ray


# TODOs
# 1. Ray doesn't have a public weakref API, but we want ray job stop to kill a task/remove the job and have the
# flexmanager stop caring about it.

# Design:
# 1. FlexManager maintains a list of JobRequests with a list of min slices and max slices.
# 2. A JobDriver spins up with a job (consisting of a callable, min and and max slices, ...)
# 3. JobDriver sends a JobRequest to FlexManager.
# 4. FlexManager attempts to create a slice for every requested slice.
# 5. As slices become available, it reserves them for JobDrivers in order of allocation (with some policy and min/max)
# 6. JobDriver periodically polls [1] FlexManager


# [1] We use polling b/c there are no public weak refs in Ray and we want the job driver to die whenever




################################################################################
# Helper dataclasses
################################################################################

@dataclass(slots=True)
class JobRequest:
    job_id: str
    min_slices: int
    max_slices: int
    slice_type: str
    driver: "ray.actor.ActorHandle"  # Actor of JobDriver
    runtime_env: dict = field(default_factory=dict)



@ray.remote(num_cpus=0.1)
class FlexSliceManager:
    """
    A generalization-ish of RayTpuManager to support workloads
    """

    # TODO: figure out weak refs...


    def __init__(self, topology):
        self.jobs: dict[ActorHandle, JobRequest] = []

    def register_job(self, job_req: JobRequest):
        self.jobs.append(job_req)





@dataclass(slots=True)
class SliceRef:
    slice_id: str
    tpu_name: str
    num_hosts: int
    is_sim: bool = False

################################################################################
# Worker task – one‑shot process executes the cmd_callable in a fresh env
################################################################################

@ray.remote(max_calls=1)
def _worker_task(cmd_callable: Callable[[], object]) -> object:
    return cmd_callable()

################################################################################
# TPUBroker – FIFO queue per TPU type
################################################################################

@ray.remote(num_cpus=0.1)
class TPUBroker:
    def __init__(
        self,
        slice_type: str,
        simulate: bool = False,
        sim_slices: int = 0,
    ):
        self.slice_type = slice_type
        self.simulate = simulate
        self.sim_slices = sim_slices
        self.free_slices: deque[SliceRef] = deque(self._discover())
        self.wait_q: deque[JobRequest] = deque()

    def request_slices(self, req: JobRequest) -> None:
        self.wait_q.append(req)
        self._dispatch()

    def release_slices(self, slices: List[SliceRef]) -> None:
        self.free_slices.extend(slices)
        self._dispatch()

    def _dispatch(self) -> None:
        while self.wait_q and len(self.free_slices) >= self.wait_q[0].min_slices:
            req = self.wait_q.popleft()
            n = min(req.max_slices, len(self.free_slices))
            granted = [self.free_slices.popleft() for _ in range(n)]
            req.driver.grant_slices.remote(granted, req)

    def _discover(self) -> List[SliceRef]:
        if self.simulate:
            return [SliceRef(f"FAKE-slice-{i}", f"fake-{i}", 1, True)
                    for i in range(self.sim_slices)]
        from ray_tpu import RayTpuManager
        mgr = RayTpuManager()
        head_res = f"TPU-{self.slice_type}-head"
        total = int(ray.cluster_resources().get(head_res, 0))
        if total <= 0:
            return []
        tpu_infos = mgr.reserve(self.slice_type, total)
        return [SliceRef(f"TPU-{info.name}-head", info.name, info.num_hosts, False)
                for info in tpu_infos]

################################################################################
# SliceActor – owns fan‑out worker tasks
################################################################################

@ray.remote
class SliceActor:
    def __init__(self, sref: SliceRef):
        self.sref = sref

    def run_job(
        self,
        jid: str,
        cmd: Callable[[], object],
        runtime_env: dict,
    ) -> List[object]:
        opts: Dict = {"runtime_env": runtime_env}
        if not self.sref.is_sim:
            opts["resources"] = {self.sref.tpu_name: 1, "TPU": self.sref.num_hosts}
        tasks = [_worker_task.options(**opts).remote(cmd)
                 for _ in range(self.sref.num_hosts)]
        return ray.get(tasks)

################################################################################
# JobDriver – orchestrates one training job
################################################################################

@ray.remote
class JobDriver:
    def __init__(
        self,
        min_slices: int,
        max_slices: int,
        slice_type: str,
        train_entrypoint: Callable[[], object],
        runtime_env: Optional[dict] = None,
        simulate: bool = False,
        sim_slices: int = 0,
    ):
        self.min = min_slices
        self.max = max_slices
        self.slice_type = slice_type
        self.fn = train_entrypoint
        self.runtime_env = runtime_env or {}
        self.simulate = simulate
        self.sim_slices = sim_slices
        self._done = False
        self.slice_actors: Dict[str, ray.actor.ActorHandle] = {}

        name = f"broker-{slice_type}"
        try:
            self.broker = ray.get_actor(name)
        except ValueError:
            self.broker = TPUBroker.options(name=name).remote(
                slice_type, simulate, sim_slices)
        self._request_slices()

    def grant_slices(self, slices: List[SliceRef], req: JobRequest) -> None:
        outs = []
        for sref in slices:
            if sref.slice_id not in self.slice_actors:
                actor = SliceActor.options(name=sref.slice_id).remote(sref)
                self.slice_actors[sref.slice_id] = actor
            else:
                actor = self.slice_actors[sref.slice_id]
            outs.append(actor.run_job.remote(req.job_id, self.fn, req.runtime_env))
        for idx, ref in enumerate(outs):
            result = ray.get(ref)
            print(f"[driver] result from {slices[idx].slice_id}: {result}")
        self._done = True

    @ray.method(enable_task_events=False)
    def is_done(self, wait_for: float = 1.0) -> str:
        if self._done:
            return "done"
        time.sleep(wait_for)
        return "pending"

    def _request_slices(self) -> None:
        jid = str(uuid.uuid4())
        req = JobRequest(
            jid, self.min, self.max, self.slice_type,
            current_actor_handle(), self.runtime_env)
        self.broker.request_slices.remote(req)

################################################################################
# CLI demo
################################################################################

def _dummy_train() -> str:
    print("Running training step …")
    return "train-ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min", type=int, default=2)
    p.add_argument("--max", type=int, default=3)
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--sim_slices", type=int, default=4)
    args = p.parse_args()

    ray.init()

    driver = JobDriver.remote(
        args.min, args.max,
        "FAKE" if args.simulate else "v4-32",
        _dummy_train,
        {"pip": ["ray"]},
        args.simulate, args.sim_slices,
    )

    while ray.get(driver.is_done.remote(0.1)) == "pending":
        time.sleep(1)


if __name__ == "__main__":
    main()
