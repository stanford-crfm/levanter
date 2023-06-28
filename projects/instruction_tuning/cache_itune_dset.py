import json

import datasets
import tqdm
import zstandard as zstd


ds1 = datasets.load_dataset("Muennighoff/P3", split="train", streaming=True)
ds2 = datasets.load_dataset("Muennighoff/natural-instructions", split="train", streaming=True)
ds2 = ds2.map(lambda x: {"inputs": "\n".join([x["definition"], x["inputs"]]), "targets": x["targets"]})
ds = datasets.concatenate_datasets([ds1, ds2])

compressor = zstd.ZstdCompressor(level=9)

with zstd.open("itune-train.jsonl.zstd", "w", cctx=compressor) as out:
    for ex in tqdm.tqdm(ds):
        out.write(json.dumps(ex) + "\n")
