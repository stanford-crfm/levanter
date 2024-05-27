import fire
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from analysis.wandb_metrics import get_marin_olmo_metrics_mapping

DATA_DIR = "scratch/data"
OUTPUT_DIR = "scratch/output"
OLMO_1B_TRAIN_DATA_PATH = f"{DATA_DIR}/OLMo-1B_train.csv"
OLMO_1B_EVAL_DATA_PATH = f"{DATA_DIR}/OLMo-1B_eval.csv"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)


def plot_comparison(
    df_marin: pd.DataFrame, df_olmo: pd.DataFrame, marin_key: str, olmo_key: str, run_id: str, note: str = ""
):
    # get proper limit to y-axis by taking the min and max after running for 5000 steps
    df_marin_limit = df_marin[df_marin["_step"] >= 5000]
    df_olmo_limit = df_olmo[df_olmo["_step"] >= 5000]
    min_val = min(df_marin_limit[marin_key].min(), df_olmo_limit[olmo_key].min()) * 0.9
    max_val = max(df_marin_limit[marin_key].max(), df_olmo_limit[olmo_key].max()) * 1.1

    plt.figure(figsize=(10, 6))
    plt.plot(df_marin["_step"], df_marin[marin_key], label="Marin")
    plt.plot(df_olmo["_step"], df_olmo[olmo_key], label="OLMo-1B")
    plt.xlabel("Step")
    plt.ylabel(marin_key)
    plt.ylim(min_val, max_val)
    plt.legend()
    title = f"OLMo-1B ({olmo_key}) vs Marin ({marin_key})"
    if note:
        title += f" ({note})"
    plt.title(title)
    marin_key = marin_key.replace("/", "_")
    olmo_key = olmo_key.replace("/", "_")
    out_name = f"run_{run_id}_olmo_1b_{marin_key}_{olmo_key}"
    if note:
        note = note.replace(" ", "_")
        out_name += f"_{note}"
    out_file = f"{OUTPUT_DIR}/{out_name}.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    plt.close()


def compare_marin_to_olmo(marin_run_id: str = "eo302w0523"):
    marin_train_data_path = f"{DATA_DIR}/marin_{marin_run_id}_train.csv"
    marin_eval_data_path = f"{DATA_DIR}/marin_{marin_run_id}_eval.csv"
    for path in [marin_train_data_path, marin_eval_data_path, OLMO_1B_TRAIN_DATA_PATH, OLMO_1B_EVAL_DATA_PATH]:
        assert Path(path).exists(), f"File not found at {path}"

    df_marin_train = pd.read_csv(marin_train_data_path)
    df_marin_eval = pd.read_csv(marin_eval_data_path)
    df_olmo_train = pd.read_csv(OLMO_1B_TRAIN_DATA_PATH)
    df_olmo_eval = pd.read_csv(OLMO_1B_EVAL_DATA_PATH)

    # Limit OLMo-1B data to the same steps as Marin data
    max_step = df_marin_train["_step"].max()
    print(f"Limiting OLMo-1B data to steps <= {max_step}")
    df_olmo_train = df_olmo_train[df_olmo_train["_step"] <= max_step]
    df_olmo_eval = df_olmo_eval[df_olmo_eval["_step"] <= max_step]

    # compare metrics
    metrics_mapping = get_marin_olmo_metrics_mapping()
    for marin_key, olmo_key in metrics_mapping.items():
        df_marin = df_marin_train if "train" in marin_key else df_marin_eval
        df_olmo = df_olmo_train if "train" in olmo_key else df_olmo_eval
        if marin_key not in df_marin:
            print(f"Missing key {marin_key} in Marin data")
            continue
        if olmo_key not in df_olmo:
            print(f"Missing key {olmo_key} in OLMo-1B data")
            continue
        if len(df_marin[marin_key].unique()) == 1:
            print(f"Skipping {marin_key} with only one unique value")
            continue

        plot_comparison(df_marin, df_olmo, marin_key, olmo_key, marin_run_id)


def compare_marin_to_olmo_train_loss(marin_run_id: str = "eo302w0523", smooth: bool = False, window_size: int = 256):
    if smooth:
        marin_train_data_path = f"{DATA_DIR}/marin_{marin_run_id}_train_smooth_{window_size}.csv"
    else:
        marin_train_data_path = f"{DATA_DIR}/marin_{marin_run_id}_train.csv"
    for path in [marin_train_data_path, OLMO_1B_TRAIN_DATA_PATH]:
        assert Path(path).exists(), f"File not found at {path}"
    df_marin_train = pd.read_csv(marin_train_data_path)
    df_olmo_train = pd.read_csv(OLMO_1B_TRAIN_DATA_PATH)
    # Limit OLMo-1B data to the same steps as Marin data
    max_step = df_marin_train["_step"].max()
    print(f"Limiting OLMo-1B data to steps <= {max_step}")
    df_olmo_train = df_olmo_train[df_olmo_train["_step"] <= max_step]

    # compare metrics
    metrics_mapping = get_marin_olmo_metrics_mapping()
    for marin_key, olmo_key in metrics_mapping.items():
        if "train" not in marin_key:
            continue
        df_marin = df_marin_train
        df_olmo = df_olmo_train
        if marin_key not in df_marin:
            print(f"Missing key {marin_key} in Marin data")
            continue
        if olmo_key not in df_olmo:
            print(f"Missing key {olmo_key} in OLMo-1B data")
            continue
        if len(df_marin[marin_key].unique()) == 1:
            print(f"Skipping {marin_key} with only one unique value")
            continue

        note = ""
        if smooth:
            note = f"smoothed with window size {window_size}"
        plot_comparison(df_marin, df_olmo, marin_key, olmo_key, marin_run_id, note)


if __name__ == "__main__":
    # fire.Fire(compare_marin_to_olmo)
    fire.Fire(compare_marin_to_olmo_train_loss)
