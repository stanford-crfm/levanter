import fire
import ray
from levanter.config import main as config_main
from levanter.main.train_lm import main as train_lm_main


ray.init()


@ray.remote
def train_lm(config_path: str):
    config_main(train_lm_main)(args=[config_path])


if __name__ == "__main__":
    fire.Fire(train_lm)
