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
    accelerator_type="v4-128",
    num_slices=2,
    with_mxla=True
)
class Test:
    def __init__(self, a: str):
        self._a = a

    def print(self):
        print(self._a)

    def test(self):
        import jax
        return jax.device_count()


a = Test()
ray.get(a.test())
