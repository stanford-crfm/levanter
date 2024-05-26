from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh


def local_device_grid_positions(mesh, process_index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns a tuple of nd arrays, one for each axis, indicating the position of each device on the grid.
    Analogous to what np.where would return."""
    if process_index is None:
        process_index = jax.process_index()

    my_device_pos = np.vectorize(lambda dev: dev.process_index == process_index)(mesh.devices)
    return my_device_pos.nonzero()


def local_devices_mapping(mesh: Mesh, process_index: Optional[int] = None) -> dict[int, int]:
    """
    Handles the case when different devices in same process share the same data in TP.
    Returns a mapping from local devices' DP/FSDP group index in global mesh to local indices
    """
    local_device_pos = local_device_grid_positions(mesh, process_index)[:2]  # first 2 axes are DP axes.
    result = {}
    uid = 0
    for local_device_index in range(len(local_device_pos[0])):
        key = local_device_pos[0][local_device_index] * mesh.devices.shape[1] + local_device_pos[1][local_device_index]
        if key not in result:
            # when two devices maps to the same key (different TP index), they will get the same data
            result[key] = uid
            uid += 1
    return result


def process_mesh_mapping(mesh) -> dict[int, int]:
    """
    Handles the case when different processes share the same data in TP.
    If we envision each process as a subgrid of the mesh for its devices, this is the position of the process
    in the coarsened process-level mesh
    """
    devices = mesh.devices
    result = {}
    uid = 0
    leftmost2uid = {}
    # basic logic: process index -> upper-left device -> TP index 0 device -> process index -> uid
    for process_index in range(jax.process_count()):
        tmp = [np.min(axis) for axis in local_device_grid_positions(mesh, process_index)]
        tmp[-1] = 0  # we want the device with TP group index 0 in the same DP/FSDP group
        upper_left_position = tuple(tmp)  # in order to index into devices
        upper_left_process = devices[upper_left_position].process_index
        # assign uid to each process that has a device with TP group index 0
        if upper_left_process not in leftmost2uid:
            leftmost2uid[upper_left_process] = uid
            uid += 1
        this_uid = leftmost2uid[upper_left_process]
        result[process_index] = this_uid

    return result
