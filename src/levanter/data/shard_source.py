import json
import os
from typing import Callable, Generic, Iterator, List, Sequence, TypeVar

import datasets
import fsspec

from levanter.utils import fsspec_utils


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)


class ShardedDataSource(Generic[T_co]):
    """
    A ShardedDataSource is the main interface for reading data. It's basically a mapping from shard names to iterators,
    with the extra feature that it exposes the ability to skip to a particular row in a shard.
    """

    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def num_shards(self) -> int:
        return len(self.shard_names)

    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        return self.open_shard_at_row(shard_name, 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T_co]:
        raise NotImplementedError

    def iter_data(self):
        """
        Iterate over all data in the dataset, in order.
        """
        for shard_name in self.shard_names:
            for doc in self.open_shard(shard_name):
                yield doc

    def map(self, fn: Callable[[T_co], T]) -> "ShardedDataSource[T]":
        return MappedShardedDataSource(self, fn)


class MappedShardedDataSource(ShardedDataSource[T]):
    def __init__(self, source: ShardedDataSource[T_co], fn: Callable[[T_co], T]):
        self.source = source
        self.fn = fn

    @property
    def shard_names(self) -> Sequence[str]:
        return self.source.shard_names

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        return map(self.fn, self.source.open_shard_at_row(shard_name, row))


class HFDatasetDataSource(ShardedDataSource[dict]):
    """
    This class is responsible for loading a dataset from HuggingFace Datasets and returning the shards.
    Only (some) IterableDatasets are actually sharded in any meaningful way, so we just return a single shard
    for all other datasets.

    kwargs are passed to load_dataset
    """

    def __init__(self, id, *, split, **kwargs):
        self.id = id
        self.split = split
        self.kwargs = kwargs
        self._shard_names = self._compute_shard_names()

    @property
    def shard_names(self) -> Sequence[str]:
        return self._shard_names

    def _compute_shard_names(self):
        dataset = self._load_dataset()
        if isinstance(dataset, datasets.IterableDataset):
            try:
                return [str(i) for i in range(dataset.n_shards)]
            except NotImplementedError:
                return ["data"]
        else:
            return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        dataset = self._load_dataset()
        if isinstance(dataset, datasets.IterableDataset) and shard_name != "data":
            # ex_iterable has a key that gets discarded typically
            shard = map(lambda t: t[1], dataset._ex_iterable.shard_data_sources([int(shard_name)]))
        else:
            shard = dataset

        idx = 0
        for doc in shard:
            if idx >= row:
                yield doc
            idx += 1

    def _load_dataset(self):
        # obnoxiously, the dataset loading stuff doesn't work with ray because of multiprocessing
        # so we have to do this hacky thing where we load the dataset in the worker
        return datasets.load_dataset(self.id, split=self.split, **self.kwargs)


class JsonlDataSource(ShardedDataSource[dict]):
    def __init__(self, urls):
        self.urls = urls
        self._shard_name_to_url_mapping = _mk_shard_name_mapping(urls)

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            # TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
            # which is not nothing, but not ideal.
            for line in f.readlines():
                if i >= row:
                    yield json.loads(line)
                i += 1


class TextDataSource(ShardedDataSource[dict]):
    def __init__(self, urls):
        self.urls = urls
        self._shard_name_to_url_mapping = _mk_shard_name_mapping(urls)

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            for line in f.readlines():
                yield line


class JsonDataSource(ShardedDataSource[dict]):
    def __init__(self, urls):
        self.urls = urls
        self._shard_name_to_url_mapping = _mk_shard_name_mapping(urls)

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        with fsspec.open(url, "r", compression="infer") as f:
            # TODO: would be nice if we could seek faster than this. Can't even skip json parsing
            data = json.load(f)
            return iter(data[row:])


def _mk_shard_name_mapping(urls):
    _shard_name_to_url_mapping = {}
    # remove common prefix
    if len(urls) == 1:
        common_prefix = os.path.dirname(urls[0])
    else:
        common_prefix = os.path.commonprefix(urls)

    missing_urls: List[str] = []

    for url in urls:
        if not fsspec_utils.exists(url):
            missing_urls.append(url)
            continue
        # escape the url for the shard name
        shard_name = url
        if common_prefix:
            shard_name = url[len(common_prefix) :]
            if shard_name.startswith("/"):
                shard_name = shard_name[1:]

        shard_name = shard_name.replace(".", "_")
        _shard_name_to_url_mapping[shard_name] = url

    if missing_urls:
        # format nicely
        missing_urls_str = "\n  - ".join(missing_urls)
        raise FileNotFoundError(f"Could not find the following urls:\n  - {missing_urls_str}")

    return _shard_name_to_url_mapping
