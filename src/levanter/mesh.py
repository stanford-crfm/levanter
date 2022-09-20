from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import jax
import numpy as np
from jax.experimental.maps import Mesh

from haliax.partitioning import ResourceAxis


@dataclass
class MeshInfo:
    mesh: Mesh
    batch_size: int
    per_device_parallelism: int  # how many inputs we can execute per device

    process_index: int = jax.process_index()
    process_count: int = jax.process_count()
    devices_per_node: int = jax.local_device_count()

    @property
    def axis_names(self):
        return self.mesh.axis_names

    @cached_property
    def model_axis_size(self):
        return self.mesh.shape[self.model_axis_name]

    @cached_property
    def data_axis_size(self):
        return self.mesh.shape[self.data_axis_name]

    @property
    def data_axis_name(self):
        return ResourceAxis.DATA

    @property
    def model_axis_name(self):
        return ResourceAxis.MODEL

    @property
    def local_data_axis_size(self):
        """number of groups of devices on this node processing distinct examples"""
        local_device_count = jax.local_device_count()
        local_model_size = self.local_model_axis_size
        assert local_device_count % local_model_size == 0
        return local_device_count // local_model_size

    @property
    def local_model_axis_size(self):
        """size of the model axis for devices on this node. This is local_device_count if the model axis size exceeds
        the number of devices on this node."""
        local_device_count = jax.local_device_count()
        if local_device_count <= self.model_axis_size:
            return local_device_count
        else:
            assert local_device_count % self.model_axis_size == 0
            return self.model_axis_size

    @cached_property
    def local_device_grid_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a tuple of nd arrays, one for each axis, indicating the position of each device on the grid.
        Analogous to what np.where would return."""
        pi = self.process_index
        # our device indices are [process_index * num_devices_per_node, (process_index + 1) * num_devices_per_node)
        # we could be clever here and do math to figure out where we are in the grid, but it's simpler and less
        # fragile to just search the grid for our devices
        my_device_pos = np.vectorize(lambda dev: dev.process_index == pi)(self.mesh.devices)
        return my_device_pos.nonzero()

    @property
    def process_mesh_position(self) -> Tuple[int, int]:
        """
        If we envision each process as a subgrid of the mesh for its devices, this is the position of the process
        in the coarsened process-level mesh
        """
        upper_left_position = np.array([np.min(axis) for axis in self.local_device_grid_positions])
        local_mesh_size = self.mesh.local_mesh.devices.shape
        pos = upper_left_position // local_mesh_size
        assert len(pos) == 2
        return pos[0], pos[1]

    @property
    def process_mesh_size(self) -> Tuple[int, int]:
        """
        If we envision each process as a subgrid of the mesh for its devices, then there is a process grid that
        is a coarsened version of the mesh. This is the size of the process grid.
        """
        local_mesh_size = self.mesh.local_mesh.devices.shape
        assert self.data_axis_size % local_mesh_size[0] == 0
        assert self.model_axis_size % local_mesh_size[1] == 0
        return (
            self.data_axis_size // local_mesh_size[0],
            self.model_axis_size // local_mesh_size[1],
        )

    @property
    def microbatch_size(self):
        """number of examples in a microbatch, across all nodes"""
        return self.data_axis_size * self.per_device_parallelism

    @property
    def per_device_batch_size(self):
        """number of examples processed by a device for an entire batch"""
        if self.batch_size is None:
            return self.per_device_parallelism

        assert self.batch_size % self.data_axis_size == 0
        ret = self.batch_size // self.data_axis_size

        return ret

    @property
    def microbatches_per_step(self):
        if self.batch_size is None:
            return 1

        assert self.batch_size % self.microbatch_size == 0
        return self.batch_size // self.microbatch_size
