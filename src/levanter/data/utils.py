from typing import Iterable, Iterator, List, TypeVar


T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yields batches of the given size from the given iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


def safe_enumerate(iterator: Iterator[T]) -> Iterator[T]:
    index = 0
    while True:
        try:
            item = next(iterator)
            yield item
            index += 1
        except StopIteration:
            break
        except Exception as e:
            print(f"Error on item {index}: {e}")
            index += 1
            continue
