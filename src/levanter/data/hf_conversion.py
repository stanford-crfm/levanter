# converts a huggingface dataset to a new huggingface dataset that has a simple, regular format of sharded,
# compressed jsonl files
# this functionality arises from repeated frustration with huggingface datasets' performance, lack of sharding etc.
# It's still a huggingface dataset, so it can be used with the huggingface datasets library, but it's also a
# simple format that can be used in other ways
import argparse
import datetime
import json
import os

import datasets
import jinja2
import tqdm
import zstandard as zstd
from datasets import Dataset, DatasetDict, DatasetInfo, Split


_my_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace dataset to another HF dataset compatible with Sprucfluo. Mainly for LM data"
    )
    parser.add_argument("--dataset", type=str, help="dataset to convert")
    parser.add_argument("--name", type=str, default=None, help="dataset name to convert")
    parser.add_argument("--output", type=str, required=True, help="output directory")
    parser.add_argument("--num_shards", type=int, default=1, help="number of shards to split the (train) dataset into")
    parser.add_argument("--level", type=int, default=9, help="compression level")
    parser.add_argument("--upload", action="store_true", help="upload the dataset to huggingface", default=False)

    args = parser.parse_args()

    name = args.name
    dataset_id = args.dataset
    dataset: DatasetDict = datasets.load_dataset(dataset_id, name=name)
    output = args.output
    os.makedirs(output, exist_ok=True)

    orig_info: DatasetInfo = next(iter(dataset.values())).info
    orig_description = orig_info.description
    description = f"Automatically generated on {datetime.datetime.now()} for {dataset_id}"
    if name:
        description += f" ({name})"
    if args.num_shards > 1:
        description += f", split into {args.num_shards} shards"
    description += "."
    description += f"\n\nOriginal Description:\n{orig_description}"

    metadata = {
        "name": f"{dataset_id}-{name}",
        "splits": {},
    }

    # Step 1: generate data
    for split, data in dataset.items():
        num_shards = args.num_shards if split == Split.TRAIN else 1
        out_dir = os.path.join(output, f"data/{split}")
        os.makedirs(out_dir, exist_ok=True)

        split_metadata = write_split(split, data, output, f"data/{split}", num_shards, args.level)

        metadata["splits"][split] = split_metadata

    # Step 2: format and write template
    template = open(os.path.join(_my_dir, "dataset.py.template"), "r").read()
    with open(os.path.join(output, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    str_info = f"""DatasetInfo(
            description={repr(description)},
            citation={repr(orig_info.citation)},
            homepage={repr(orig_info.homepage)},
            license={repr(orig_info.license)},
            version="{orig_info.version}",
            features=Features.from_dict({repr(orig_info.features.to_dict())}),
            supervised_keys={repr(orig_info.supervised_keys)})"""

    dataset_module = jinja2.Template(template).render(info=str_info)
    with open(os.path.join(output, "dataset.py"), "w") as f:
        f.write(dataset_module)

    if args.upload:
        # Step 3: upload to huggingface
        import huggingface_hub

        repo_name = os.path.basename(output)
        user = huggingface_hub.whoami()["name"]
        repo = f"{user}/{repo_name}"
        huggingface_hub.create_repo(repo_name, repo_type="dataset")
        err = os.system(f"cd {output} && git init")
        if not err:
            err = os.system(f"cd {output} && git remote add origin https://huggingface.co/datasets/{repo}")
        if not err:
            err = os.system(f"cd {output} && git fetch origin")
        if not err:
            err = os.system(f"cd {output} && git reset --hard origin/main")
        # if not err: err = os.system(f"cd {args.output} && git rebase origin/main")
        if not err:
            err = os.system(f'cd {output} && git lfs track "*.zst"')
        if not err:
            err = os.system(f"cd {output} && git add -A && git commit -m 'init' && git push origin main:main")

        if not err:
            print(f"Dataset uploaded to https://huggingface.co/datasets/{user}/{repo_name}")

            # Now write urls in a way that sprucfluo can read them
            url_base = f"https://huggingface.co/datasets/{repo}/resolve/main"
            final_urls = {
                split: [f"{url_base}/{file}" for file in split_metadata["files"]]
                for split, split_metadata in metadata["splits"].items()
            }

            print(json.dumps(final_urls, indent=2))
            with open(os.path.join(output, "urls.json"), "w") as f:
                json.dump(final_urls, f, indent=2)


def write_split(split, dataset: Dataset, base_dir, out_dir, processor, num_shards, compression_level):
    split_metadata = {}
    if num_shards == 1:
        all_files = [f"{out_dir}/{split}.jsonl.zst"]
    else:
        all_files = [f"{out_dir}/{split}_{i}_of_{num_shards}.jsonl.zst" for i in range(num_shards)]

    opened = [
        zstd.open(os.path.join(base_dir, f), mode="w", cctx=zstd.ZstdCompressor(level=compression_level))
        for f in all_files
    ]

    try:
        for i, item in enumerate(tqdm.tqdm(dataset, desc=f"Writing {split}")):
            if processor:
                item = processor(item)
            if len(item["text"]) == 0:
                continue
            shard = i % num_shards
            opened[shard].write(json.dumps(item) + "\n")
    finally:
        for f in opened:
            f.close()
    split_metadata["files"] = all_files
    return split_metadata


if __name__ == "__main__":
    main()
