import fire
import jax
import ray

from levanter.config import main as config_main
from levanter.main.train_lm import main as train_lm_main


ray.init()
print(f"ray.available_resources(): {ray.available_resources()}")
print(f"jax.device_count(): {jax.device_count()}")

@ray.remote(resources={"TPU": 8})
def train_lm(config_path: str):
    config_main(train_lm_main)(args=[config_path])


def main(config_path: str):
    train_lm.remote(config_path)

if __name__ == "__main__":
    fire.Fire(main)
