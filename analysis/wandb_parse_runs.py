from pathlib import Path
from typing import List
import fire
import pandas as pd
import wandb
from tqdm import tqdm
from wandb_metrics import get_olmo_metrics_keys, get_marin_metrics_keys


ENTITY_OLMO = "ai2-llm"
ENTITY_MARIN = "stanford-mercury"
PROJECT_OLMO_1B = "OLMo-1B"
PROJECT_MARIN = "marin"
N_SAMPLES = 10000000  # set it to be large enough to get all the data
TIMEOUT = 3600  # seconds
OUT_DIR = "scratch/data"
Path(OUT_DIR).mkdir(exist_ok=True, parents=True)


def check_missing_steps(df: pd.DataFrame, step_size: int = 1000):
    steps = df["_step"].values
    step_diffs = steps[1:] - steps[:-1]
    missing_steps = step_diffs[step_diffs != step_size]
    print(f"Missing steps: {missing_steps}")


def get_all_olmo_runs(project_name: str = PROJECT_OLMO_1B, target: str = "eval"):
    # we need to extract train/loss and eval/loss separately; otherwise, the train loss will be sampled
    if target == "eval":
        keys = get_olmo_metrics_keys()
    elif target == "train":
        keys = ["train/CrossEntropyLoss"]
    else:
        raise ValueError(f"Unknown target: {target}")

    api = wandb.Api(timeout=TIMEOUT)
    path = f"{ENTITY_OLMO}/{project_name}"
    runs = api.runs(path=path, per_page=1000)
    dfs = []
    for _, run in tqdm(enumerate(runs)):
        df = run.history(samples=N_SAMPLES, keys=keys)
        if len(df) > 0:
            dfs.append(df)

    # merge all runs
    df_all = pd.concat(dfs)
    df_all.sort_values("_step", inplace=True)
    max_step = df_all["_step"].max()
    min_step = df_all["_step"].min()

    # check for steps
    print(f"Found {len(dfs)} runs for {target} with {df_all.shape[0]} rows, steps: {min_step} - {max_step}")
    step_size = 1000 if target == "eval" else 1
    check_missing_steps(df_all, step_size=step_size)

    # save to file
    out_file = f"{OUT_DIR}/{project_name}_{target}.csv"
    print(f"Saving {df_all.shape[0]} rows to {out_file}")
    df_all.to_csv(out_file, index=False)


def smooth_column(df: pd.DataFrame, cols: List[str], window_size: int = 256):
    """Smooth a column with a rolling average"""
    for col in cols:
        df[col] = df[col].rolling(window=window_size, center=True).mean()
    return df


def get_marin_run(run_id: str, target="eval", smooth: bool = False, window_size: int = 256):
    if target == "eval":
        keys = get_marin_metrics_keys()
    elif target == "train":
        keys = ["train/loss"]
    api = wandb.Api(timeout=TIMEOUT)
    path = f"{ENTITY_MARIN}/{PROJECT_MARIN}"
    run = api.run(f"{path}/{run_id}")
    df_history = run.history(samples=N_SAMPLES, keys=keys)
    if smooth and target == "train":
        df_history = smooth_column(df_history, keys, window_size=window_size)

    out_name = f"marin_{run_id}_{target}"
    if smooth:
        out_name += f"_smooth_{window_size}"
    out_file = f"{OUT_DIR}/{out_name}.csv"
    print(f"Saving {df_history.shape[0]} rows to {out_file}")
    df_history.to_csv(out_file, index=False)


if __name__ == "__main__":
    # get_all_olmo_runs(target="train")
    # get_all_olmo_runs(target="eval")
    fire.Fire(get_marin_run)
