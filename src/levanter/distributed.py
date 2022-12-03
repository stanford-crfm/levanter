import os
from typing import List, Optional

import jax


# This is a copy-paste and modification of the original SlurmCluster class in jax.
# The key difference is that it uses the SLURM_LOCAL_PROCESS_COUNT to determine how many devices to use

_JOBID_PARAM = "SLURM_JOB_ID"
_NODE_LIST = "SLURM_STEP_NODELIST"
_PROCESS_COUNT = "SLURM_NTASKS"
_PROCESS_ID = "SLURM_PROCID"
_LOCAL_PROCESS_ID = "SLURM_LOCALID"
_NUM_NODES = "SLURM_STEP_NUM_NODES"
_TASKS_PER_NODE = "SLURM_TASKS_PER_NODE"
_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


class LevanterSlurmCluster:
    @classmethod
    def is_env_present(cls) -> bool:
        return _JOBID_PARAM in os.environ

    @classmethod
    def get_local_process_id(cls) -> Optional[int]:
        return int(os.environ[_LOCAL_PROCESS_ID])

    @classmethod
    def get_local_device_ids_for_process(cls) -> Optional[List[int]]:
        if _VISIBLE_DEVICES in os.environ:
            all_visible_devices = [int(x) for x in os.environ[_VISIBLE_DEVICES].split(",")]
        else:
            all_visible_devices = list(range(jax.local_device_count()))

        num_local_processes = int(os.environ[_TASKS_PER_NODE])
        local_process_id = cls.get_local_process_id()

        if local_process_id is None:
            return None

        if len(all_visible_devices) % num_local_processes != 0:
            raise ValueError(
                f"Number of visible devices ({len(all_visible_devices)}) is not divisible by the number "
                f"of local processes ({num_local_processes})"
            )

        num_devices_per_local_process = len(all_visible_devices) // num_local_processes

        # select contiguous devices for this process
        begin = local_process_id * num_devices_per_local_process
        return all_visible_devices[begin : begin + num_devices_per_local_process]
