from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh


def local_device_grid_positions(mesh, process_index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of nd arrays, one for each axis, indicating the position of each device on the grid.
    Analogous to what np.where would return."""
    pi = process_index or jax.process_index()
    # our device indices are [process_index * num_devices_per_node, (process_index + 1) * num_devices_per_node)
    # we could be clever here and do math to figure out where we are in the grid, but it's simpler and less
    # fragile to just search the grid for our devices
    my_device_pos = np.vectorize(lambda dev: dev.process_index == pi)(mesh.devices)
    return my_device_pos.nonzero()


def local_devices_mapping(mesh: Mesh, process_index: Optional[int] = None) -> dict[int, int]:
    """Returns a mapping from local devices' DP/FSDP group index in global mesh to local indices"""
    local_device_pos = local_device_grid_positions(mesh, process_index)[:2]  # first 2 axes are DP axes.
    result = {}
    j = 0
    for i in range(len(local_device_pos[0])):
        key = local_device_pos[0][i] * mesh.devices.shape[1] + local_device_pos[1][i]
        if key not in result:
            result[key] = j  # in case of TP=2, local device 0 and 2 will be mapped to same key.
            j += 1
    return result


def process_mesh_mapping(mesh) -> dict[int, int]:
    """
    If we envision each process as a subgrid of the mesh for its devices, this is the position of the process
    in the coarsened process-level mesh
    """
    devices = mesh.devices
    result = {}  # maps process index to leftmost process index in DP/FSDP group
    i = 0
    leftmost2uid = {}

    for i in range(jax.process_count()):
        tmp = [np.min(axis) for axis in local_device_grid_positions(mesh, i)]
        tmp[-1] = 0  # we want the device with TP group index 0 in the same DP/FSDP group
        upper_left_position = tuple(tmp)  # in order to index into devices
        upper_left_process = devices[upper_left_position].process_index
        # assign uid to each process that has a device with TP group index 0
        if upper_left_process not in leftmost2uid:
            leftmost2uid[upper_left_process] = i
            i += 1
        uid = leftmost2uid[upper_left_process]
        result[i] = uid

    return result
