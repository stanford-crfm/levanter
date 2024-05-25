import fire
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from analysis.wandb_metrics import get_marin_olmo_metrics_mapping

DATA_DIR = "scratch/data"
OUTPUT_DIR = "scratch/output"
OLMO_1B_DATA_PATH = f"{DATA_DIR}/OLMo-1B.csv"
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)


def compare_marin_to_olmo(marin_run_id: str = "eo302w0523"):
    marin_data_path = f"{DATA_DIR}/marin_{marin_run_id}.csv"
    assert Path(marin_data_path).exists(), f"Marin data not found at {marin_data_path}"
    assert Path(OLMO_1B_DATA_PATH).exists(), f"OLMo-1B data not found at {OLMO_1B_DATA_PATH}"
    df_marin = pd.read_csv(marin_data_path)
    df_olmo = pd.read_csv(OLMO_1B_DATA_PATH)

    # Limit OLMo-1B data to the same steps as Marin data
    max_step = df_marin["_step"].max()
    print(f"Limiting OLMo-1B data to steps <= {max_step}")
    df_olmo = df_olmo[df_olmo["_step"] <= max_step]

    # compare metrics
    metrics_mapping = get_marin_olmo_metrics_mapping()
    for marin_key, olmo_key in metrics_mapping.items():
        if marin_key not in df_marin:
            print(f"Missing key {marin_key} in Marin data")
            continue
        if olmo_key not in df_olmo:
            print(f"Missing key {olmo_key} in OLMo-1B data")
            continue
        if len(df_marin[marin_key].unique()) == 1:
            print(f"Skipping {marin_key} with only one unique value")
            continue

        # get proper limit to y-axis by taking the min and max after running for 5000 steps
        df_marin_limit = df_marin[df_marin["_step"] >= 5000]
        df_olmo_limit = df_olmo[df_olmo["_step"] >= 5000]
        min_val = min(df_marin_limit[marin_key].min(), df_olmo_limit[olmo_key].min()) * 0.9
        max_val = max(df_marin_limit[marin_key].max(), df_olmo_limit[olmo_key].max()) * 1.1

        plt.figure(figsize=(10, 6))
        plt.plot(df_marin["_step"], df_marin[marin_key], label=f"Marin: {marin_run_id}")
        plt.plot(df_olmo["_step"], df_olmo[olmo_key], label="OLMo-1B")
        plt.xlabel("Step")
        plt.ylabel(marin_key)
        plt.ylim(min_val, max_val)
        plt.legend()
        plt.title(f"Marin ({marin_key}) vs OLMo-1B ({olmo_key})")
        marin_key = marin_key.replace("/", "_")
        olmo_key = olmo_key.replace("/", "_")
        out_file = f"{OUTPUT_DIR}/run_{marin_run_id}_olmo_1b_{marin_key}_{olmo_key}.png"
        plt.savefig(out_file)
        print(f"Saved plot to {out_file}")
        plt.close()


if __name__ == "__main__":
    fire.Fire(compare_marin_to_olmo)
