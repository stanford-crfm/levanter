#!/usr/bin/env python
"""Benchmark cross-slice weight transfer using ``jax.experimental.transfer``.

A persistent large TPU slice hosts the weights while small slices
periodically connect to fetch them. Process 0 of the large slice runs a
:class:`TransferServer` and coordinates a broadcast/all-gather cycle to
simulate parameter updates.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp
import wandb
from jax.experimental import multihost_utils
from jax.experimental import transfer as jax_transfer
import ray

from levanter.infra import ray_tpu
from ray.actor import ActorHandle


@dataclass
class TransferStats:
    """Simple container for timing results."""

    bytes_transferred: int
    latency_s: float


@ray.remote
class AddressHolder:
    """Stores the weight server address for other actors."""

    def __init__(self, **wandb_args) -> None:
        self.address: str | None = None
        wandb.init(**wandb_args)

    def set(self, addr: str) -> None:
        self.address = addr

    def get(self) -> str | None:
        return self.address

    def log(self, metrics):
        wandb.log(metrics)


@jax.jit
def deshard(arr):
    """Deshard the array to a single device."""
    # TODO: right now we insist it fit on a single device. ideally it would be up to X devices
    # using a process mesh
    return jax.lax.with_sharding_constraint(arr, jax.sharding.PartitionSpec())



def large_slice_loop(size: int, rounds: int, holder: ActorHandle) -> None:
    """Run on the large slice; serves weights and simulates updates."""

    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("d",))
    with jax.sharding.use_mesh(mesh):
        @jax.jit
        def make_arr(key):
            """Create a random array of the given size."""
            out = jax.random.normal(key, (size,), dtype=jnp.float32)
            return jax.lax.with_sharding_constraint(out, jax.sharding.PartitionSpec("d",))
        arr = make_arr(jax.random.PRNGKey(0))

        server = None
        if jax.process_index() == 0:
            server = jax_transfer.start_transfer_server(jax.devices()[0].client)
            ray.get(holder.set.remote(server.address()))

        for step in range(rounds):
            multihost_utils.broadcast_one_to_all(
                jnp.array(step, dtype=jnp.int32),
                is_source=jax.process_index() == 0,
            )
            if jax.process_index() == 0 and server is not None:

                time_in = time.time()
                gathered = deshard(arr)
                gathered = jax.block_until_ready(gathered)
                gather_elapsed = time.time() - time_in

                time_in = time.time()
                server.await_pull(step, gathered)
                pull_elapsed = time.time() - time_in

                nbytes = gathered.nbytes
                del gathered
                metrics = {
                    "host/gather_time": gather_elapsed,
                    "hot/pull_time": pull_elapsed,
                    "bytes_transferred": nbytes,
                    "host/step": step,
                }
                ray.get(holder.log.remote(metrics))


def small_slice_worker(server_holder: ActorHandle, uuid: int, shape: Iterable[int]) -> TransferStats:
    """Connect to the weight server, pull the parameters, and place them on this slice's TPUs."""

    # connect to the remote transfer server from the small slice
    client_server = jax_transfer.start_transfer_server(jax.devices()[0].client)
    connection = client_server.connect(server_address)
    server_address = ray.get(server_holder.get.remote())

    mesh = jax.make_mesh((jax.device_count(),), ("d",))
    with jax.sharding.use_mesh(mesh):
        # create a placeholder for the weights to be pulled into
        # placeholder = jnp.zeros(shape, dtype=jnp.float32)
        @jax.jit
        def make_placeholder():
            """Create a placeholder array with the given shape."""
            out = jnp.zeros(shape, dtype=jnp.float32)
            return jax.lax.with_sharding_constraint(out, jax.sharding.PartitionSpec("d",))

        placeholder = make_placeholder()

        start = time.time()
        result = connection.pull(uuid, placeholder)
        pull_complete = time.time()

        @jax.jit
        def reshard(arr):
            """Reshard the array to the current device."""
            return jax.lax.with_sharding_constraint(arr, jax.sharding.PartitionSpec("d",))

        result = reshard(result)
        jax.block_until_ready(result)

        elapsed = time.time() - start

        # log the transfer stats
        metrics = {
            "client/pull_time": pull_complete - start,
            "client/reshard_time": elapsed - (pull_complete - start),
            "bytes_transferred": result.nbytes,
            "host/uuid": uuid,
        }

        ray.get(server_holder.log.remote(metrics))

        return TransferStats(bytes_transferred=result.nbytes, latency_s=elapsed)


def run_benchmark(
    large_type: str, small_type: str, size: int, num_small: int, rounds: int
) -> list[TransferStats]:
    holder: ActorHandle = AddressHolder.remote()  # type: ignore[attr-defined]

    large_future = ray_tpu.run_on_pod_ray.remote(
        # ray doesn't support partial
        # partial(large_slice_loop, size, rounds, holder),
        lambda : large_slice_loop(size, rounds, holder),
        large_type,
    )

    server_address = None
    while server_address is None:
        server_address = ray.get(holder.get.remote())
        if server_address is None:
            time.sleep(1.0)

    shape = (size,)
    results: list[TransferStats] = []
    for i in range(rounds):
        workers = [
            ray_tpu.run_on_pod_ray.remote(
                lambda : small_slice_worker(holder, i, shape),
                small_type,
            )
            for _ in range(num_small)
        ]
        results.extend(ray.get(workers))

    ray.get(large_future)
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
    for i, s in enumerate(stats):
        print(f"Transfer {i}: {s.bytes_transferred / 1e6:.2f} MB in {s.latency_s:.2f}s")


if __name__ == "__main__":
    main()
