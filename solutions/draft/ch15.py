"""Reference answer: ordered bounded inference batches."""
from collections.abc import Callable, Iterable


def batched_predict(values: Iterable, predictor: Callable, batch_size: int = 8) -> list:
    """Call predictor once per bounded batch and preserve one result per item."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output, pending = [], []
    for value in values:
        pending.append(value)
        if len(pending) == batch_size:
            result = list(predictor(pending))
            if len(result) != len(pending):
                raise ValueError("predictor must return one result per input")
            output.extend(result)
            pending = []
    if pending:
        result = list(predictor(pending))
        if len(result) != len(pending):
            raise ValueError("predictor must return one result per input")
        output.extend(result)
    return output
