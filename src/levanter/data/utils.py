from typing import TypeVar, Iterable, Iterator, List

T = TypeVar('T')


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
