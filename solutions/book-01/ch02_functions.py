"""Separate reference answers for Chapter 2's progressive preprocessing labs.

These standard-library-only functions are independent of notebook state.
Attempt the learner exercises before comparing with these answers.
"""
from collections.abc import Iterable


def text_features(text: str) -> list[float]:
    """Return [word count, mean word length] using lowercase whitespace words.

    Punctuation remains part of each word. Empty text has two zero features.
    Non-string text raises TypeError; the original string is never changed.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    words = text.lower().split()
    if not words:
        return [0.0, 0.0]
    total_length = 0
    for word in words:
        total_length += len(word)
    return [float(len(words)), float(total_length / len(words))]


def prepare_sample(record: dict, label_to_id: dict[str, int]) -> dict:
    """Create exactly id/features/label_id without modifying either input.

    Required fields and the supplied label mapping are accessed directly:
    missing fields and unknown labels raise KeyError. Text errors propagate.
    The caller supplies the record and a string-to-integer label mapping.
    """
    return {
        "id": record["id"],
        "features": text_features(record["text"]),
        "label_id": label_to_id[record["label"]],
    }


def make_batch(
    records: Iterable[dict], label_to_id: dict[str, int]
) -> tuple[list[list[float]], list[int]]:
    """Consume records once and return aligned feature rows and label IDs.

    Each row is independently created through prepare_sample. Empty input
    returns ([], []). Source and preparation errors propagate, potentially
    after partial consumption of an iterator; input records are not edited.
    """
    features = []
    label_ids = []
    for record in records:
        sample = prepare_sample(record, label_to_id)
        features.append(sample["features"])
        label_ids.append(sample["label_id"])
    return features, label_ids
