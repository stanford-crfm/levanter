import json
import os
import warnings
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, List, Optional, Sequence, Sized, TypeVar

import datasets
import fsspec

from levanter.utils import fsspec_utils

from ._preprocessor import (
    BatchResult,
    _BatchMapTransform,
    _construct_composite_batch_processor,
    _DatasetTransform,
    _MapTransform,
)
from .dataset import Dataset, ShardableDataset
from .utils import batched


if TYPE_CHECKING:
    from levanter.data.shard_cache import MetricsMonitor


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")


class ShardedDataset(Dataset[T_co]):
    """
    A ShardedDataset is the main interface for reading data. It's basically a mapping from shard names to iterators,
    with the extra feature that it exposes the ability to skip to a particular row in a shard.

    The difference between a [ShardedDataset][] and a [ShardableDataset][] is that a [ShardedDataset][]
    has a fixed number of shards, and a [ShardableDataset][] supports a `shard` method that can be used to
    split the dataset into multiple shards.
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

    def __iter__(self):
        """
        Iterate over all data in the dataset, in order.
        """
        for shard_name in self.shard_names:
            for doc in self.open_shard(shard_name):
                yield doc

    def build_cache(
        self,
        path: str,
        *,
        rows_per_chunk: Optional[int] = None,
        await_finished: bool = True,
        monitors: Optional[Sequence["MetricsMonitor"]] = None,
    ) -> ShardableDataset[dict]:
        """
        Constructs a shard cache version of this dataset using Ray.

        Levanter's preprocessing pipeline offers the following features/guarantees:
        * distributed, sharded preprocessing using Ray
        * deterministic ordering of data
        * interruptible and resumable
        * streaming results (no need to wait for everything to finish)

        *Note that build_cache does not in general preserve the order of the data.*

        Note that this is an experimental API and is subject to change. It is also not very well tested, so use at your
        own risk.

        Returns:
            A new dataset that is backed by the cache.
        """
        from levanter.data.shard_cache import DEFAULT_ROWS_PER_CHUNK, DictCacheDataset, build_cache

        if rows_per_chunk is None:
            rows_per_chunk = DEFAULT_ROWS_PER_CHUNK

        source, processor = _construct_composite_batch_processor(self)

        cache = build_cache(
            path, source, processor, rows_per_chunk=rows_per_chunk, await_finished=await_finished, monitors=monitors
        )
        return DictCacheDataset(cache)

    def map(self, fn: Callable[[T_co], U]) -> "ShardedDataset[U]":
        return _MappedShardedDataset(self, fn)

    def map_batches(
        self, fn: Callable[[list[T_co]], BatchResult], batch_size, *, num_cpus=1, num_gpus=0, **resources
    ) -> "ShardedDataset[dict]":
        """
        **Lazily** map a function over batches of data. This is useful for doing things like batching data for a model,
        or for batched preprocessing.

        This function is **lazy**.

        Args:
            fn:  A function that takes a list of data and returns an iterable of results
            batch_size: The batch size to use
            num_cpus: passed to ray
            num_gpus: passed to ray
            **resources: Resources to pass to Ray

        Returns:
            A new ShardedDataset.
        """
        return _BatchMappedShardedDataset(self, fn, batch_size, num_cpus=num_cpus, num_gpus=num_gpus, **resources)


def dataset_from_hf(id: str, *, split, **kwargs) -> ShardedDataset[dict]:
    """
    Create a ShardedDataset from a HuggingFace dataset. Arguments are passed to load_dataset.
    """
    return WrappedHFDataset(id, split=split, **kwargs)


def dataset_from_jsonl(urls_or_paths: Sequence[str]) -> ShardedDataset[dict]:
    return JsonlDataset(urls_or_paths)


class WrappedHFDataset(ShardedDataset[dict]):
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


class TextUrlDataset(ShardedDataset[str]):
    """
    Dataset for various text formats.
    """

    def __init__(self, urls, text_key="text"):
        self.urls = urls
        self._shard_name_to_url_mapping = _mk_shard_name_mapping(urls)
        self.text_key = text_key

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[str]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            format = _sniff_format(url)
            match format:
                case ".jsonl":
                    # TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
                    # which is not nothing, but not ideal.
                    for line in f:
                        if i >= row:
                            yield json.loads(line)[self.text_key]
                        i += 1
                case ".txt":
                    for line in f:
                        if i >= row:
                            yield line
                        i += 1
                case ".json":
                    data = json.load(f)
                    for doc in data[row:]:
                        yield doc[self.text_key]
                case _:
                    raise ValueError(f"Unknown format {format}")


def _sniff_format(url):
    # should take into account compression etc
    good_formats = [".jsonl", ".txt", ".json"]
    # try both with and without compression (could be gz, bz2, etc, so look at the "first" extension)
    extensions = [os.path.splitext(url)[1], os.path.splitext(os.path.splitext(url)[0])[1]]
    for format in good_formats:
        if any(ext == format for ext in extensions):
            return format


class JsonlDataset(ShardedDataset[dict]):
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
            for line in f:
                print(i, line)
                if i >= row:
                    yield json.loads(line)
                i += 1


class TextDataset(ShardedDataset[dict]):
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
            for line in f:
                if i >= row:
                    yield line
                i += 1


class JsonDataset(ShardedDataset[dict]):
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


class _TransformedDataset:
    source: ShardedDataset
    _transform: _DatasetTransform


class _MappedShardedDataset(ShardedDataset[T], _TransformedDataset):
    def __init__(self, source: ShardedDataset[T_co], fn: Callable[[T_co], T]):
        self.source = source
        self.fn = fn
        self._transform = _MapTransform(fn)

    @property
    def shard_names(self) -> Sequence[str]:
        return self.source.shard_names

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        return map(self.fn, self.source.open_shard_at_row(shard_name, row))


class _BatchMappedShardedDataset(ShardedDataset[T], _TransformedDataset):
    def __init__(
        self,
        source: ShardedDataset[T_co],
        fn: Callable[[list[T_co]], Iterable[U]],
        batch_size,
        num_cpus=1,
        num_gpus=0,
        **resources,
    ):
        self.source = source
        self._transform = _BatchMapTransform(fn, batch_size, num_cpus, num_gpus, resources)

    @property
    def shard_names(self) -> Sequence[str]:
        return self.source.shard_names

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        warnings.warn("This is not the best way to use batched preprocessing. Use build_cache instead.")
        # this one is tricky because we have to do batching ourselves and there's no guarantee that input and output
        # batch sizes are the same
        i = 0
        shard_iter = self.source.open_shard_at_row(shard_name, row)
        for batch in batched(shard_iter, self._transform.batch_size):  # type: ignore
            result = self._transform.fn(batch)  # type: ignore
            if isinstance(result, Sized) and len(result) + i < row:
                i += len(result)
                continue

            for doc in result:
                if i >= row:
                    yield doc
                i += 1
