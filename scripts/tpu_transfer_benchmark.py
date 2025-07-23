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
from functools import partial
from typing import Iterable

import jax
import jax.numpy as jnp
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

    def __init__(self) -> None:
        self.address: str | None = None

    def set(self, addr: str) -> None:
        self.address = addr

    def get(self) -> str | None:
        return self.address


def _all_gather(weights: jax.Array) -> jax.Array:
    @partial(jax.pmap, axis_name="d")
    def gather(x):
        return jax.lax.all_gather(x, "d", tiled=True)

    return gather(weights)


def large_slice_loop(size: int, rounds: int, holder: ActorHandle) -> None:
    """Run on the large slice; serves weights and simulates updates."""

    num_devices = jax.device_count()
    per_device = size // num_devices
    key = jax.random.key(0)
    arr = jax.random.normal(key, (num_devices, per_device), dtype=jnp.float32)
    weights = jax.device_put_sharded(list(arr), jax.devices())

    server = None
    if jax.process_index() == 0:
        server = jax_transfer.start_transfer_server(jax.devices()[0].client)
        ray.get(holder.set.remote(server.address()))

    for step in range(rounds):
        multihost_utils.broadcast_one_to_all(
            jnp.array(step, dtype=jnp.int32),
            is_source=jax.process_index() == 0,
        )
        gathered = _all_gather(weights)
        if jax.process_index() == 0 and server is not None:
            server.await_pull(step, gathered)


@ray.remote(max_calls=1)
def small_slice_worker(server_address: str, uuid: int, shape: Iterable[int]) -> TransferStats:
    """Connect to the weight server, pull the parameters, and place them on this slice's TPUs."""

    # connect to the remote transfer server from the small slice
    client_server = jax_transfer.start_transfer_server(jax.devices()[0].client)
    connection = client_server.connect(server_address)

    placeholder = jax.ShapeDtypeStruct(shape=tuple(shape), dtype=jnp.float32)

    start = time.time()
    result = connection.pull(uuid, placeholder)

    # shard the weights across the devices of the small slice so each device
    # holds a distinct slice.
    num_devices = jax.device_count()
    shards = jnp.reshape(result, (num_devices, -1))
    sharded = jax.device_put_sharded([shards[i] for i in range(num_devices)], jax.devices())
    jax.block_until_ready(sharded)

    elapsed = time.time() - start
    return TransferStats(bytes_transferred=result.nbytes, latency_s=elapsed)


def run_benchmark(
    large_type: str, small_type: str, size: int, num_small: int, rounds: int
) -> list[TransferStats]:
    holder: ActorHandle = AddressHolder.remote()  # type: ignore[attr-defined]

    large_future = ray_tpu.run_on_pod_ray.remote(
        partial(large_slice_loop, size, rounds, holder),
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
                partial(small_slice_worker, server_address, i, shape),
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
    parser.add_argument("--size", type=int, default=int(32e9), help="Number of fp32 weights")
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
