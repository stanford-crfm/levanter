import os
import re
import subprocess
from typing import List, Optional

import jax


_JOBID_PARAM = "SLURM_JOB_ID"
_NODE_LIST_CHOICES = ["SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST", "SLURM_NODELIST"]
_PROCESS_COUNT = "SLURM_NTASKS"
_PROCESS_ID = "SLURM_PROCID"
_LOCAL_PROCESS_ID = "SLURM_LOCALID"
_NUM_NODES = "SLURM_STEP_NUM_NODES"
_TASKS_PER_NODE = "SLURM_STEP_TASKS_PER_NODE"
_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
_NODE_NAME = "SLURM_TOPOLOGY_ADDR"


class LevanterSlurmCluster:
    """
    This class is a copy-paste and modification of the original SlurmCluster class in jax, with a few differences:
    - It uses the SLURM_LOCAL_PROCESS_COUNT to determine how many devices to use
    - It looks in a few places for the node list, since the environment variable is set differently
    depending on how you run
    # TODO: upstream this
    """

    @classmethod
    def is_env_present(cls) -> bool:
        return _JOBID_PARAM in os.environ

    @classmethod
    def get_local_process_id(cls) -> Optional[int]:
        return int(os.environ[_LOCAL_PROCESS_ID])

    # this is mostly copy paste, but it looks at a different env variable that is set when sbatch is set
    # TODO: upstream this
    @classmethod
    def get_coordinator_address(cls) -> str:
        # Pick port in ephemeral range [(65535 - 2^12 + 1), 65535]
        port = int(os.environ[_JOBID_PARAM]) % 2**12 + (65535 - 2**12 + 1)

        # Parse the first hostname of the job
        # If we are looking for 'node001',
        # node_list potential formats are 'node001', 'node001,host2',
        # 'node[001-0015],host2', and 'node[001,007-015],host2'.
        node_list = LevanterSlurmCluster._node_list()
        if node_list is None:
            raise ValueError(
                "Could not find node list in environment variables. You must set coordinator_address manually."
            )
        delims = {",", "["}
        ind = next((i for i, ch in enumerate(node_list) if ch in delims), len(node_list))
        if ind == len(node_list) or node_list[ind] == ",":  # Formats: 'node001' or 'node001,host2'
            return f"{node_list[:ind]}:{port}"
        else:  # Formats: 'node[001-0015],host2' or 'node[001,007-015],host2'
            prefix = node_list[:ind]
            suffix = node_list[ind + 1 :]
            delims2 = {",", "-"}
            ind2 = next((i for i, ch in enumerate(suffix) if ch in delims2), None)
            return f"{prefix}{suffix[:ind2]}:{port}"

    @classmethod
    def _node_list(cls):
        return next((os.environ[o] for o in _NODE_LIST_CHOICES if o in os.environ), None)

    @classmethod
    def get_local_device_ids_for_process(cls) -> Optional[List[int]]:
        if _VISIBLE_DEVICES in os.environ:
            all_visible_devices = [int(x) for x in os.environ[_VISIBLE_DEVICES].split(",")]
        else:
            all_visible_devices = list(range(jax.local_device_count()))

        # We want to figure out how many tasks are running on this node
        # the only env variable that is reliably set here is SLURM_STEP_TASKS_PER_NODE
        # which is a comma separated list of the number of tasks per node, except they "helpfully"
        # collapse the list if there are multiple nodes with the same number of tasks e.g.
        # 1(x2),3,4(x3) -> 1,1,3,4,4,4
        # Why they do this is beyond me. It seems like more trouble to be less helpful.
        # So we have to do some parsing to figure out how many tasks are on each node
        # and then figure out which node we are on
        # first replace the repeated values with the number of times they are repeated
        unrolled_tasks_per_node = []

        multi_match = re.compile(r"(\d+)\(x(\d+)\)")
        for x in os.environ[_TASKS_PER_NODE].split(","):
            match = multi_match.match(x)
            if match:
                unrolled_tasks_per_node.extend([int(match.group(1))] * int(match.group(2)))
            else:
                unrolled_tasks_per_node.append(int(x))

        # now we can figure out which node we are on. This is also annoying because the node list
        # is a comma separated list of nodes, but they collapse the list if there are multiple nodes
        # with the same name e.g. node001,node002,node003,node004,node007 -> node[001-004,007]
        # thankfully slurm exposes a command to expand this list for us
        node_list = LevanterSlurmCluster._node_list()
        if node_list is None:
            raise ValueError(
                "Could not find node list in environment variables. You must set coordinator_address manually."
            )

        node_list = (
            subprocess.check_output(["scontrol", "show", "hostnames", node_list], input=b"")
            .decode("utf-8")
            .splitlines()
        )

        # finally, we can figure out which node we are on
        local_node = os.environ[_NODE_NAME]
        local_node_index = node_list.index(local_node)
        tasks_on_local_node = unrolled_tasks_per_node[local_node_index]

        local_process_id = cls.get_local_process_id()

        if local_process_id is None:
            return None

        if len(all_visible_devices) % tasks_on_local_node != 0:
            raise ValueError(
                f"Number of visible devices ({len(all_visible_devices)}) is not divisible by the number "
                f"of local tasks ({tasks_on_local_node})"
            )

        num_devices_per_local_process = len(all_visible_devices) // tasks_on_local_node

        # select contiguous devices for this process
        begin = local_process_id * num_devices_per_local_process
        return all_visible_devices[begin : begin + num_devices_per_local_process]
