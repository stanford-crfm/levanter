import pyrallis

from levanter.data import CachedLMDatasetConfig


@pyrallis.wrap()
def main(args: CachedLMDatasetConfig):
    """Caches two different kinds of datasets. It can cache a dataset from a list of urls, or a dataset from a hf dataset"""

    for split in ["train", "validation"]:
        # TODO: think about doing this on apache beam or something fancy. Maybe nothing fancy we can do for HF datasets,
        # but for pure-url based ones, shouldn't be hard.
        args.build_or_load_document_cache(split)


if __name__ == "__main__":
    main()
