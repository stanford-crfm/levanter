# runs train_lm in a wandb sweep
import os

import draccus
import wandb

import levanter.config
from levanter.main.train_lm import TrainLmConfig
from levanter.main.train_lm import main as train_lm_main


project = os.environ.get("WANDB_PROJECT")
sweep_id = os.environ.get("WANDB_SWEEP_ID")

if project is None:
    print("No WANDB_PROJECT found in environment, exiting")
    exit(1)


if sweep_id is None:
    print("No WANDB_SWEEP_ID found in environment, exiting")
    exit(1)


def merge_in_config(config_obj: TrainLmConfig, config_delta):
    import mergedeep
    import yaml

    base_yaml = draccus.dump(config_obj)
    base_dict = yaml.safe_load(base_yaml)
    out_dict = mergedeep.merge({}, base_dict, config_delta)
    new_config = draccus.decode(TrainLmConfig, out_dict)
    return new_config


def run_levanter(base_config: TrainLmConfig):
    # tuner = wandb.controller(sweep_id, project=project)
    # config = tuner.sweep_config
    # del tuner

    # they aren't very forthcoming about which keys are defined, so we have to parse them out of the config ourselves
    def sweep_wandb_config(run_config):
        # if "parameters" in sweep_config:
        #     return {k: sweep_wandb_config(v, run_config[k]) for k, v in sweep_config["parameters"].items()}
        # else:
        return run_config

    run = wandb.init(project=project)

    assert run is not None

    sweep_run_config = dict(sweep_wandb_config(run.config))
    print(sweep_run_config)

    new_config = merge_in_config(base_config, sweep_run_config)
    print(new_config)

    train_lm_main(new_config)


def main(config: TrainLmConfig):
    wandb.agent(sweep_id, count=20, function=lambda: run_levanter(config))


if __name__ == "__main__":
    levanter.config.main(main)()
