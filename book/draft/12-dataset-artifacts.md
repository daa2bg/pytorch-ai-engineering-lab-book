# Study 12 — Version the data transformation

**First draft.** Save the prepared data and the decisions that produced it. Prerequisites: tokenization, file ingestion, and joins. The artifact contract runs with the standard library; the Hugging Face Dataset section runs when the local `datasets` package is installed.

## A saved table is only part of an experiment

Two runs can read the same documents and still train on different inputs. A changed lowercase rule, category mapping, truncation limit, or split changes the experiment. A useful **dataset artifact** therefore includes prepared rows, source IDs, split membership, and preprocessing metadata.

A **fingerprint** identifies a particular content/configuration combination. It helps detect change; it does not explain the change or prove that the data is correct. Store readable metadata as well as a hash. Record package versions when library behavior matters, and preserve the raw source separately.

Do not choose a fresh random split every time a downstream script loads data. Persist the actual IDs used for training and validation. For grouped records, such as multiple documents from one customer, split using the group identifier rather than assuming independent documents.

## Make split membership explicit

The following deterministic demonstration assigns IDs by a hash bucket. Its purpose is repeatability under row reordering. It does not guarantee balanced classes or independent groups, which must be checked separately.

```python
import hashlib

rows = [
    {"id": "a", "text": "Invoice due", "label": 1},
    {"id": "b", "text": "Claim received", "label": 0},
    {"id": "c", "text": "Payment received", "label": 1},
]
def split_name(source_id):
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 5
    return "validation" if bucket == 0 else "train"

split_ids = {"train": [], "validation": []}
for row in rows:
    split_ids[split_name(row["id"])].append(row["id"])
assert set(split_ids["train"]).isdisjoint(split_ids["validation"])
assert set(split_ids["train"] + split_ids["validation"]) == {"a", "b", "c"}
assert split_name("a") == split_name("a")
print(split_ids)
```

We hash the ID, not the target label. Sorting or appending rows does not move an existing ID to a new partition. However, changing the ID or bucket rule does. A tiny fixture may have an empty partition; real preparation must inspect counts, label coverage, and any temporal/group constraints before training.

## Use a local Hugging Face Dataset

Hugging Face Datasets handles structured collections, transformations, caching, and disk storage. The tokenizer converts text into model-specific inputs. Accelerate helps coordinate execution across devices/processes; it does not decide your data schema or split policy. These tools have separate responsibilities.

The example constructs data locally, so no Hub download or login is needed. `map` receives each record and returns fields to add or replace. With `batched=True`, the function instead receives lists per field and must return compatible-length lists; start with the simpler row contract.

```python
import importlib.util
from tempfile import TemporaryDirectory
from pathlib import Path

if importlib.util.find_spec("datasets") is None:
    print("SKIP Hugging Face Dataset: datasets is not installed.")
else:
    from datasets import Dataset, load_from_disk
    dataset = Dataset.from_list(rows)
    prepared = dataset.map(lambda row: {
        "clean_text": row["text"].lower(),
        "word_count": len(row["text"].split()),
    })
    assert prepared["id"] == ["a", "b", "c"]
    assert prepared["word_count"] == [2, 2, 2]
    with TemporaryDirectory() as folder:
        location = Path(folder) / "prepared"
        prepared.save_to_disk(str(location))
        restored = load_from_disk(str(location))
        assert restored.to_dict() == prepared.to_dict()
    print("Local Dataset transformation and round trip passed.")
```

A lambda is a short function expression; use a named function when preprocessing needs validation or several steps. A library cache fingerprint can speed repeated work, but a readable, versioned preprocessing contract should still accompany the training artifact.

## Save the contract beside the rows

```python
import json

contract = {
    "schema_version": 1,
    "lowercase": True,
    "feature_order": ["word_count"],
    "label_to_id": {"claim": 0, "invoice": 1},
    "split_ids": split_ids,
}
encoded = json.dumps({"rows": rows, "contract": contract},
                     ensure_ascii=False, sort_keys=True, allow_nan=False)
decoded = json.loads(encoded)
assert decoded["contract"]["label_to_id"]["invoice"] == 1
assert decoded["rows"][0]["id"] == "a"
print("Artifact JSON bytes:", len(encoded.encode("utf-8")))
```

This schema is small enough to inspect. It should grow only when a new decision affects reproducibility. A content hash is not encryption or anonymization; never treat hashes as permission to expose sensitive data or identifiers.

## Essential exercise — fingerprint rows and settings together

Implement `dataset_fingerprint(records, config)`. Accept JSON-compatible values supplied by the caller and return the SHA-256 hexadecimal digest of `{"records": records, "config": config}` serialized as UTF-8 JSON with sorted keys, no extra separator spaces, `ensure_ascii=False`, and `allow_nan=False`.

Dictionary insertion order must not change the result. Row order, content, or configuration changes should change it. Preserve inputs. Let JSON errors propagate, including rejection of NaN/infinity. The exact serialization contract matters when comparing results.

```python exercise
def dataset_fingerprint(records, config):
    raise NotImplementedError("Serialize canonically, encode, then hash")

first = dataset_fingerprint([{"id": "a", "text": "claim"}], {"lowercase": True})
second = dataset_fingerprint([{"text": "claim", "id": "a"}], {"lowercase": True})
assert first == second
assert len(first) == 64
```

**Optional study:** persist label/group statistics and a transformation version. Explain how an intentional preprocessing revision should invalidate a cached training result.

**Completion gate:** show retained IDs, saved split membership, a Dataset round trip when available, and a changed fingerprint after changing preprocessing.

Primary sources: https://huggingface.co/docs/datasets/create_dataset , https://huggingface.co/docs/datasets/process , and https://huggingface.co/docs/accelerate/quicktour . Reference answer: `solutions/draft/ch12.py`; tests: `tests/draft/test_ch12.py`. A missing Datasets dependency is reported as a skip, not a completed integration.
