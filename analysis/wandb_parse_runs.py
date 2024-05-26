from pathlib import Path

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


def get_all_olmo_runs(project_name: str = PROJECT_OLMO_1B):
    api = wandb.Api(timeout=TIMEOUT)
    path = f"{ENTITY_OLMO}/{project_name}"
    runs = api.runs(path=path, per_page=1000)
    olmo_keys = get_olmo_metrics_keys()
    dfs = []
    for _, run in tqdm(enumerate(runs)):
        df_history = run.history(samples=N_SAMPLES, keys=olmo_keys)
        if len(df_history) == 0:
            print(f"Skipping run {run.id} with no data")
            continue
        dfs.append(df_history)

    # merge all runs
    df_all = pd.concat(dfs)
    df_all.sort_values("_step", inplace=True)
    max_step = df_all["_step"].max()
    min_step = df_all["_step"].min()

    # check for steps
    print(f"Found {len(dfs)} runs with {df_all.shape[0]} rows, steps: {min_step} - {max_step}")
    check_missing_steps(df_all)

    # save to file
    out_file = f"{OUT_DIR}/{project_name}.csv"
    print(f"Saving {df_all.shape[0]} rows to {out_file}")
    df_all.to_csv(out_file, index=False)


def smooth_column(df: pd.DataFrame, col: str, window_size: int = 256):
    """Smooth a column with a rolling average"""
    df[col] = df[col].rolling(window=window_size, center=True).mean()
    return df


def get_marin_run(run_id: str):
    api = wandb.Api(timeout=TIMEOUT)
    path = f"{ENTITY_MARIN}/{PROJECT_MARIN}"
    run = api.run(f"{path}/{run_id}")
    keys = get_marin_metrics_keys()
    df_history = run.history(samples=N_SAMPLES, keys=keys)
    # df_history = run.history(samples=N_SAMPLES)
    # df_history = smooth_column(df_history, "train/loss", window_size=512)
    
    out_file = f"{OUT_DIR}/marin_{run_id}.csv"
    print(f"Saving {df_history.shape[0]} rows to {out_file}")
    df_history.to_csv(out_file, index=False)


if __name__ == "__main__":
    # get_all_olmo_runs()
    fire.Fire(get_marin_run)
