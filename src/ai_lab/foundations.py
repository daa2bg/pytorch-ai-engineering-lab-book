"""Inspectable solutions for introductory Python/AI labs."""
from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")

def batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield ordered batches; supports one-shot iterators."""
    if size <= 0:
        raise ValueError("size must be positive")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

def standardize(values: list[float]) -> list[float]:
    """Population z-scores; constant inputs become zeros."""
    if not values:
        raise ValueError("values must be nonempty")
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    if variance == 0:
        return [0.0] * len(values)
    return [(x - mean) / variance**0.5 for x in values]
