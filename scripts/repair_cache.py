import os
import os.path
from dataclasses import dataclass
from typing import List

import fsspec
import pyarrow
from tqdm import tqdm

import levanter
from levanter.data.shard_cache import LEDGER_FILE_NAME, CacheLedger, ChunkMetadata, _serialize_json_and_commit


@dataclass
class RepairCacheArgs:
    cache_path: str


@levanter.config.main()
def main(args: RepairCacheArgs):
    """Repairs a broken cache by recreating the ledger"""
    for split in ["train", "validation"]:
        # find train files in the dir, which can be in cloud
        fs = fsspec.get_fs_token_paths(args.cache_path)[0]
        paths = os.path.join(args.cache_path, split, "*.parquet")
        files = fs.glob(paths)

        # We're basically redoing this, but without the old ledger:
        chunks: List[ChunkMetadata] = []

        pbar = tqdm(files)
        total_input_ids = 0
        for file in pbar:
            file = f"gs://{file}"
            table = pyarrow.parquet.read_metadata(file)

            input_ids = 0
            for g in range(table.num_row_groups):
                input_ids += table.row_group(g).column(0).statistics.num_values

            file = file.replace(os.path.join(args.cache_path, split), "").lstrip("/")

            chunks.append(
                ChunkMetadata(
                    name=file.replace(".parquet", ""),
                    num_rows=table.num_rows,
                    field_counts={"input_ids": input_ids},
                )
            )

            total_input_ids += input_ids

            pbar.set_postfix(num_rows=table.num_rows, total_input_ids=total_input_ids)

        ledger = CacheLedger(chunks=chunks)
        _serialize_json_and_commit(os.path.join(args.cache_path, split, LEDGER_FILE_NAME), ledger)


if __name__ == "__main__":
    main()
