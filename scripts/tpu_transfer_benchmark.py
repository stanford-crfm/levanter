#!/usr/bin/env python
"""Benchmark cross-slice weight transfer using ``jax.experimental.transfer``.

A persistent large TPU slice hosts the weights while small slices
periodically connect to fetch them. Process 0 of the large slice runs a
:class:`TransferServer` and coordinates a broadcast/all-gather cycle to
simulate parameter updates.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp
import wandb
from jax.experimental import multihost_utils
from jax.experimental import transfer as jax_transfer
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes

from levanter.infra import ray_tpu
from ray.actor import ActorHandle

logger = logging.getLogger("ray")

import socket


def get_local_ip_from_hostname():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror:
        return "Could not resolve hostname to IP address."


@dataclass
class TransferStats:
    """Simple container for timing results."""

    bytes_transferred: int
    latency_s: float


@ray.remote
class AddressHolder:
    """Stores the weight server address for other actors."""

    def __init__(self, num_transfers: int, **wandb_args) -> None:
        self.address: str | None = None
        wandb.init(**wandb_args)
        self._total_transfers = num_transfers
        self._num_finished = 0

    def set(self, addr: str) -> None:
        self.address = addr

    def get(self) -> str | None:
        return self.address

    def mark_transfer_finished(self, step: int) -> None:
        """Mark a transfer as finished for logging purposes."""
        if step >= self._total_transfers:
            raise ValueError(f"Step {step} exceeds total transfers {self._total_transfers}.")

        self._num_finished += 1
        wandb.log({
            "manager/finished_transfers": self._num_finished,
        })

    def log(self, metrics):
        wandb.log(metrics)

    def finish(self):
        wandb.finish()

    def is_finished(self) -> bool:
        """Check if all transfers are finished."""
        return self._num_finished >= self._total_transfers

    # New helper to query how many transfers have completed so far.
    def num_finished(self) -> int:
        return self._num_finished


@jax.jit
def deshard(arr):
    """Deshard the array to a single device."""
    # TODO: right now we insist it fit on a single device. ideally it would be up to X devices
    # using a process mesh
    return jax.lax.with_sharding_constraint(arr, jax.sharding.PartitionSpec())


def large_slice_loop(size: int, rounds: int, num_workers: int, holder: ActorHandle) -> None:
    """Run on the large slice; serves weights and simulates updates."""

    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("d",))
    with jax.sharding.use_mesh(mesh):
        logger.info("Large slice started with %d devices", num_devices)

        @jax.jit
        def make_arr(rng_key):
            """Create a random array of the given size on the mesh."""
            out = jax.random.normal(rng_key, (size,), dtype=jnp.float32)
            return jax.lax.with_sharding_constraint(out, jax.sharding.PartitionSpec("d",))

        rng = jax.random.PRNGKey(0)

        server = None

        if jax.process_index() == 0:
            ip = get_local_ip_from_hostname()
            # Bind the transfer server explicitly.
            server = jax_transfer.start_transfer_server(
                jax.devices()[0].client,
                f"{ip}:0",  # bind to the local IP address
                [f"{ip}:0"] * num_devices,  # address hints – one per local TPU device
            )
            ray.get(holder.set.remote(server.address()))
            logger.info("Large slice started transfer server at %s", server.address())

        for step in range(rounds):
            # generate new parameters each round
            rng, subkey = jax.random.split(rng)
            arr = make_arr(subkey)
            logger.info("Step %d", step)

            # Deshard the parameters to a single device for serving.
            time_in = time.time()
            gathered = deshard(arr)
            gathered = jax.block_until_ready(gathered)
            gather_elapsed = time.time() - time_in

            if jax.process_index() == 0 and server is not None:
                logger.info("Gathered took %.2f seconds for step %d", gather_elapsed, step)

                # Schedule awaits for each worker with its unique UUID so they can pull.
                time_in = time.time()
                for wid in range(num_workers):
                    uuid = step * num_workers + wid
                    server.await_pull(uuid, gathered)

                pull_elapsed = time.time() - time_in

                # Wait until all workers have completed their pull for this round
                expected_done = (step + 1) * num_workers
                while ray.get(holder.num_finished.remote()) < expected_done:
                    time.sleep(0.5)

                nbytes = gathered.nbytes
                del gathered
                metrics = {
                    "host/gather_time": gather_elapsed,
                    "host/pull_time": pull_elapsed,
                    "host/step": step,
                    "host/gather_bytes": nbytes,
                }
                ray.get(holder.log.remote(metrics))


        logger.info("Finished registering await_pulls")
        while not ray.get(holder.is_finished.remote()):
            time.sleep(5.0)

        logger.info("All small slices finished step %d", step)


def small_slice_loop(
    server_holder: ActorHandle,
    worker_id: int,
    shape: Iterable[int],
    rounds: int,
    num_workers: int,
) -> list[TransferStats]:
    """Persistent worker that performs `rounds` pulls from the weight server.

    Returns a list of ``TransferStats`` (one per round).
    """

    ip = get_local_ip_from_hostname()

    # Local transfer server (required by Transfer API)
    client_server = jax_transfer.start_transfer_server(
        jax.devices()[0].client,
        f"{ip}:0",  # bind to the local IP address
        [f"{ip}:0"] * jax.device_count(),
    )

    # Wait for the large slice to publish its address
    server_address = None
    while server_address is None:
        server_address = ray.get(server_holder.get.remote())
        if server_address is None:
            time.sleep(0.5)

    connection = client_server.connect(server_address)
    logger.info("Small slice %d connected to %s", worker_id, server_address)

    mesh = jax.make_mesh((jax.device_count(),), ("d",))
    stats: list[TransferStats] = []

    with jax.sharding.use_mesh(mesh):
        # Really we need to make this array and not use shapedtypestruct b/c it wants a sharding.
        @jax.jit
        def make_placeholder():
            """Create a placeholder array with the given shape."""
            out = jnp.zeros(shape, dtype=jnp.float32)
            return jax.lax.with_sharding_constraint(out, jax.sharding.PartitionSpec("d",))
        placeholder = make_placeholder()

        @jax.jit
        def reshard(arr):
            return jax.lax.with_sharding_constraint(arr, jax.sharding.PartitionSpec("d",))

        for step in range(rounds):
            uuid = step * num_workers + worker_id
            logger.info("Small slice %d pulling weights for step %d", worker_id, step)
            start = time.time()
            result = connection.pull(uuid, placeholder)
            result = jax.block_until_ready(result)
            pull_complete = time.time()
            transfer_finished = server_holder.mark_transfer_finished.remote(uuid)

            result = reshard(result)
            jax.block_until_ready(result)

            elapsed = time.time() - start

            metrics = {
                "client/pull_time": pull_complete - start,
                "client/reshard_time": elapsed - (pull_complete - start),
                "client/bytes_transferred": result.nbytes,
                "client/worker_id": worker_id,
                "client/round": step,
            }
            # Asynchronously log & mark completion
            ray.get(
                [
                    server_holder.log.remote(metrics),
                    transfer_finished,  # mark transfer finished
                ]
            )

            stats.append(TransferStats(bytes_transferred=result.nbytes, latency_s=elapsed))

    logger.info("Small slice %d finished %d rounds", worker_id, rounds)
    return stats


def run_benchmark(
    large_type: str, small_type: str, size: int, num_small: int, rounds: int
) -> list[list[TransferStats]]:
    holder: ActorHandle = AddressHolder.options(  # type: ignore
        num_cpus=0,
        scheduling_strategy=schedule_on_head_node_strategy()
    ).remote(
        rounds * num_small,
        project="levanter-tpu-transfer-benchmark",
    )  # type: ignore[attr-defined]

    # Define a serializable function for the large slice.
    def _large_slice_fn():
        return large_slice_loop(size, rounds, num_small, holder)

    large_future = ray_tpu.run_on_pod_ray.remote(_large_slice_fn, large_type, auto_kill_busy_vfio=True)

    server_address = None
    while server_address is None:
        server_address = ray.get(holder.get.remote())
        if server_address is None:
            time.sleep(1.0)

    shape = (size,)
    # Launch `num_small` persistent workers that will operate across *all* rounds.
    # Helper to create a dedicated function per worker for serialization.
    def _make_worker_fn(wid: int):
        def _worker_fn():
            return small_slice_loop(holder, wid, shape, rounds, num_small)
        return _worker_fn

    worker_futures = [
        ray_tpu.run_on_pod_ray.remote(_make_worker_fn(wid), small_type, auto_kill_busy_vfio=True)
        for wid in range(num_small)
    ]

    # Gather all results. Each worker returns a list[TransferStats]. Flatten afterwards.
    results_nested: list[list[TransferStats]] = ray.get(worker_futures)
    results: list[TransferStats] = [stat for worker_stats in results_nested for stat in worker_stats]

    # wait for the large slice to finish
    ray.get(large_future)
    # finish the wandb run
    ray.get(holder.finish.remote())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-type", default="v5p-128", help="Shape of the large slice")
    parser.add_argument("--small-type", default="v5p-8", help="Shape of the small slices")
    parser.add_argument("--num-small", type=int, default=1, help="Number of small slices")
    parser.add_argument("--size", type=int, default=int(8e9), help="Number of fp32 weights")
    parser.add_argument("--rounds", type=int, default=1, help="Number of transfer rounds")
    args = parser.parse_args()

    ray.init()

    stats = run_benchmark(
        args.large_type, args.small_type, args.size, args.num_small, args.rounds
    )
    # Flatten defensively in case nested lists slipped through.
    flat_stats: list[TransferStats] = [stat for worker_stats in stats for stat in worker_stats]

    for i, s in enumerate(flat_stats):
        print(f"Transfer {i}: {s.bytes_transferred / 1e6:.2f} MB in {s.latency_s:.2f}s")

def schedule_on_head_node_strategy(soft=False) -> NodeAffinitySchedulingStrategy:
    """
    Create a scheduling strategy that targets the Ray head node.

    We do this in Marin because usually only the head node is non-preemptible,
    and some actors (e.g. StatusActor) should not be preempted.
    """

    node_id = get_head_node_id()
    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=soft)


def get_head_node_id() -> str:
    """Get the node ID of the Ray head node."""
    try:
        head = list_nodes(filters=[("is_head_node", "=", True)])[0]
        return head.node_id
    except StopIteration:
        raise RuntimeError("No head node found in the Ray cluster. Ensure the cluster is running.") from None



if __name__ == "__main__":
    main()
