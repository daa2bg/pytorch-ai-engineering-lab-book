# Study 13 — Search by vectors and evaluate the neighbors

**First draft.** Build a tiny vector index and make its ranking contract testable. Prerequisites: arrays/tensors and tokenization. Core examples use Python's standard library. The vectors below are transparent word-count fixtures, not learned semantic embeddings.

## Why represent text as a vector?

Keyword search answers whether words match. A vector represents an item as ordered numbers so we can compare items numerically. An **embedding** is a representation in such a vector space; learned text embeddings can capture relationships beyond exact word overlap, depending on their model and training.

We begin with a fixed vocabulary and counts. This lets us inspect every coordinate before using a pretrained encoder later. A value at position zero must mean the same feature for documents and queries. Reordering the vocabulary for only one side silently damages retrieval.

**Cosine similarity** compares direction rather than raw length: dot product divided by the product of vector lengths. Identical positive directions have similarity one, perpendicular directions zero, and opposite directions minus one. With nonnegative count vectors, our fixture will not produce negative similarities.

## Build a small, explicit index

```python
import math

vocabulary = ("invoice", "payment", "claim", "vehicle")
def count_vector(text):
    words = text.lower().split()
    return [float(words.count(term)) for term in vocabulary]

index = {
    "d1": count_vector("invoice payment"),
    "d2": count_vector("vehicle claim"),
    "d3": count_vector("payment invoice"),
}
assert index["d1"] == [1.0, 1.0, 0.0, 0.0]
assert count_vector("unrecognized vocabulary") == [0.0, 0.0, 0.0, 0.0]
print(index)
```

Unknown words contribute zero coordinates. The all-zero query is therefore not evidence that every document is equally relevant; it means this representation has no matching signal.

A Python dictionary maps stable source IDs to vectors. A real index also needs the representation model/version, normalization policy, and document version. If the embedding model changes, old and new vectors may no longer share a comparable space.

## Calculate and rank

```python
def demo_cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_length = math.sqrt(sum(a * a for a in left))
    right_length = math.sqrt(sum(b * b for b in right))
    if left_length == 0 or right_length == 0:
        return 0.0
    return dot / (left_length * right_length)

query = count_vector("payment")
ranked = sorted(
    ((source_id, demo_cosine(query, vector)) for source_id, vector in index.items()),
    key=lambda item: (-item[1], item[0]),
)
assert [source_id for source_id, score in ranked] == ["d1", "d3", "d2"]
assert ranked[0][1] == ranked[1][1]
top_two = ranked[:2]
assert {source_id for source_id, score in top_two} == {"d1", "d3"}
print(top_two)
```

The sort key is a tuple: negative score gives descending similarity, then source ID breaks ties deterministically. Stable tie behavior makes tests and debugging easier. `ranked[:2]` selects the first two items, giving a basic **top-k** search.

The demo cosine assumes equal widths. Python's `zip` would otherwise stop at the shorter vector and produce a plausible but invalid answer. The exercise adds explicit shape checks. Finite values matter too: NaN can corrupt ordering while being hard to recognize in a longer pipeline.

Our zero-vector policy is similarity zero. A retrieval service can then abstain or fall back to another method when the query contains no recognized signal. Returning the first ID as though it were a confident match would hide the representation failure.

## Evaluate the representation, not just the code

A test showing that cosine returns one for equal directions verifies arithmetic. It does not show that the nearest document answers a real question. Build a small labeled retrieval fixture: query, known relevant source IDs, and expected policy for no evidence.

```python
relevant = {"d1", "d3"}
retrieved = {source_id for source_id, score in top_two}
recall = len(relevant & retrieved) / len(relevant)
assert recall == 1.0
empty_query = count_vector("weather tomorrow")
assert all(demo_cosine(empty_query, vector) == 0.0 for vector in index.values())
print("Fixture recall@2:", recall, "(tiny constructed example, not a benchmark)")
```

The next study uses this separation between retrieval quality and response quality. A fluent response cannot repair a missing relevant document.

## Essential exercise — safe cosine

Implement `cosine(left, right)` for equal-width, nonempty finite real-valued sequences. Raise `ValueError` for empty vectors, unequal widths, or NaN/infinity. Return `0.0` when either vector is all zeros. Do not mutate inputs. No third-party dependency is necessary.

The small-count demo above uses direct arithmetic. For this general contract, first divide each nonzero vector by its own largest absolute coordinate. Positive scaling preserves direction, while keeping the largest magnitude at one avoids overflow when squaring huge inputs and underflow when squaring uniformly tiny inputs. Then compute the dot product and lengths of the scaled vectors. Check that `[1e308, 0]` and `[1e-300, 0]` each have similarity one with themselves.

```python exercise
def cosine(left, right):
    raise NotImplementedError("Validate widths and finite values, then normalize")

assert abs(cosine([1.0, 0.0], [2.0, 0.0]) - 1.0) < 1e-12
assert cosine([0.0, 0.0], [1.0, 2.0]) == 0.0
assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
```

**Optional study:** replace count vectors with one explicitly identified pretrained embedding model. Keep query/document preprocessing consistent and compare retrieval on the same relevance fixture. A pretrained model download is separate from running this offline study.

**Completion gate:** explain feature ordering, zero-vector policy, tie-breaking, and why arithmetic tests are different from retrieval evaluation.

Primary source for the corresponding tensor operation: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html . Reference answer: `solutions/draft/ch13.py`; tests: `tests/draft/test_ch13.py`.
