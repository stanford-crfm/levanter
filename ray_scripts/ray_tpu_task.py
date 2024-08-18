import ray
import logging
from ray_scripts import ray_tpu


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ray.init()
ray_tpu.init()

print("Cluster resources: ", ray_tpu.cluster_resources())
print("Available resources: ", ray_tpu.available_resources())


@ray_tpu.remote(
    accelerator_type="v4-8",
    num_slices=2,
    with_mxla=True,
    env={
        "TPU_STDERR_LOG_LEVEL": "0", "TPU_MIN_LOG_LEVEL": "0", "TF_CPP_MIN_LOG_LEVEL": "0"
    },
)
def test():
    import jax
    return jax.device_count()

print("Running test")
print(ray.get(test()))

ray.shutdown()
