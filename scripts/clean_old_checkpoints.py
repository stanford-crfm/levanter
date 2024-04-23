## Script to aid in cleaning up old checkpoints from a directory of runs. just prints directories that can be deleted.
## Doesn't delete the latest complete checkpoint or the latest one with step ending in '000'.
# You can pass this to gsutil to delete the directories, e.g.:
# python clean_old_checkpoints.py gs://my-bucket/my-dir | xargs -I {} gsutil -m rm -r {}
import os
import sys

import fsspec


def is_dir_of_checkpoints(path):
    fs = fsspec.filesystem("gcs")
    # if the children are named like step-XXXXX, then it's a checkpoint directory
    children = fs.ls(path)
    return any("step-" in child for child in children)


def list_deletable_directories(base_dir):
    fs = fsspec.filesystem("gcs")
    run_ids = fs.ls(base_dir)

    for run_id in run_ids:
        if not is_dir_of_checkpoints(run_id):
            continue

        checkpoint_paths = fs.ls(run_id)
        max_complete_checkpoint = None
        max_000_checkpoint = None

        # remove any paths that are not checkpoint directories (i.e. they don't have step-XXXXX in their name)
        checkpoint_paths = [path for path in checkpoint_paths if "/step-" in path]

        for path in checkpoint_paths:
            checkpoint_dir = fs.ls(path)
            if os.path.join(path, "metadata.json") in checkpoint_dir:
                # Check if this checkpoint is the largest complete checkpoint
                step_number = int(path.split("-")[-1])
                if max_complete_checkpoint is None or step_number > int(max_complete_checkpoint.split("-")[-1]):
                    max_complete_checkpoint = path

                # Check if this checkpoint ends in '000' and is the largest of such
                if str(step_number).endswith("000"):
                    if max_000_checkpoint is None or step_number > int(max_000_checkpoint.split("-")[-1]):
                        max_000_checkpoint = path

        # Add all checkpoint directories except the ones we need to keep
        for path in checkpoint_paths:
            if path != max_complete_checkpoint and path != max_000_checkpoint:
                yield path


# Usage example:
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_old_checkpoints.py <base_dir>")
        sys.exit(1)
    base_dir = sys.argv[1]

    for path in list_deletable_directories(base_dir):
        print(f"gs://{path}")
