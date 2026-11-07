# Chapter 26 — Deliver a small end-to-end ML workflow

**First integrated draft.** Bring the earlier skills together: ingest records, validate a contract, prepare features, compare a baseline, train a model, evaluate held-out examples, save artifacts, reload without retraining, and expose a prediction boundary with an operational escape hatch. The goal is an inspectable local system, not a production-quality document classifier.

This capstone uses CPU PyTorch, the standard library, and a deliberately tiny synthetic dataset. It downloads no weights and does not claim pretrained-model adaptation. Study 09 supplies the separate pretrained fine-tuning path. In a later iteration, substitute that model behind the same application boundary and compare it against the baseline on representative held-out data.

## Define data and features before optimizing

We will classify text containing the explicit words `claim` or `invoice`. Two binary features indicate those words. This is a transparent signal chosen to make the engineering path easy to test. Real correspondence is ambiguous and cannot be evaluated credibly with six invented examples.

```python
import hashlib
import io
import json
import tempfile
from pathlib import Path
import torch
from torch import nn

torch.manual_seed(19)
torch.set_num_threads(1)
label_to_id = {"claim": 0, "invoice": 1}
feature_names = ["contains_claim", "contains_invoice"]
records = [
    {"id":"a", "text":"claim received", "label":"claim"},
    {"id":"b", "text":"invoice received", "label":"invoice"},
    {"id":"c", "text":"new claim", "label":"claim"},
    {"id":"d", "text":"new invoice", "label":"invoice"},
    {"id":"e", "text":"claim pending", "label":"claim"},
    {"id":"f", "text":"invoice pending", "label":"invoice"},
]
source_text = "\n".join(json.dumps(record, sort_keys=True) for record in records)
source_digest = hashlib.sha256(source_text.encode()).hexdigest()
loaded = [json.loads(line) for line in io.StringIO(source_text)]
assert len({record["id"] for record in loaded}) == len(loaded)
assert all(record["label"] in label_to_id for record in loaded)

def features(text):
    words = set(text.lower().split())
    return [float(name in words) for name in ("claim", "invoice")]

def as_batch(rows):
    inputs = torch.tensor([features(row["text"]) for row in rows], dtype=torch.float32)
    targets = torch.tensor([label_to_id[row["label"]] for row in rows], dtype=torch.long)
    return inputs, targets

train_rows, test_rows = loaded[:4], loaded[4:]
x_train, y_train = as_batch(train_rows)
x_test, y_test = as_batch(test_rows)
baseline_id = 0  # deterministic tie-break for the balanced training labels
baseline_accuracy = (y_test == baseline_id).float().mean().item()
assert baseline_accuracy == 0.5
print("Held-out majority baseline:", baseline_accuracy)
```

The split is explicit and ID-disjoint, but feature patterns repeat across it. A high held-out score will verify the toy rule, not establish robust generalization. A real split should reflect the intended deployment boundary, for example later arrival times or unseen customers, while avoiding duplicate-document leakage.

## Train, measure, and keep the metadata

The model is a linear classifier with two input features and two output class scores. Cross-entropy trains those scores against category IDs. The optimization budget is fixed, and the final accuracy is measured rather than assumed.

```python
model = nn.Linear(2, 2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
for step in range(50):
    optimizer.zero_grad()
    loss = nn.functional.cross_entropy(model(x_train), y_train)
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    prediction = model(x_test).argmax(dim=1)
    model_accuracy = (prediction == y_test).float().mean().item()
assert model_accuracy > baseline_accuracy
print("Observed toy held-out accuracy:", model_accuracy)
```

Weights alone are not a deployable contract. Preserve feature order, label mapping, source version/digest, training configuration, and measured results. The service must apply the same preprocessing order after restart. Here we save a state dictionary, reload only weights/tensors and simple metadata, and compare outputs exactly enough for floating-point computation.

```python
with tempfile.TemporaryDirectory() as temporary:
    artifact = Path(temporary) / "model.pt"
    metadata = {
        "schema_version": 1, "source_sha256": source_digest,
        "feature_names": feature_names, "label_to_id": label_to_id,
        "seed": 19, "steps": 50, "baseline_accuracy": baseline_accuracy,
        "model_accuracy": model_accuracy,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, artifact)
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["source_sha256"] == source_digest
    assert checkpoint["metadata"]["feature_names"] == feature_names
    restored = nn.Linear(2, 2)
    restored.load_state_dict(checkpoint["state_dict"])
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x_test), model(x_test))
    print("Restart path: loaded validated artifact without running training again.")
```

Only load trusted artifacts; a `weights_only` argument is a narrower loading mode, not a general guarantee that arbitrary untrusted files are safe. In a durable service, publish weights and metadata together under a versioned artifact reference. Validate the expected schema and feature contract before serving; retrain explicitly if data or configuration fingerprints change.

## Expose a boundary that can decline

A request containing neither signal—or both signals—is outside this toy model's clean feature contract. Route it to review instead of reporting a confident answer. Keep the rule visible and testable. The following function is a local service boundary; it does not start a web server.

```python
def predict_document(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    vector = features(text)
    if sum(vector) != 1:
        return {"label": "review", "reason": "ambiguous_or_unsupported_features"}
    with torch.no_grad():
        identity = restored(torch.tensor([vector])).argmax(dim=1).item()
    inverse = {value: key for key, value in label_to_id.items()}
    return {"label": inverse[identity], "reason": "toy_model"}

assert predict_document("invoice attached")["label"] == "invoice"
assert predict_document("please help")["label"] == "review"
event = {"event": "prediction_completed", "route": "review", "model_version": "toy-v1"}
print(json.dumps(event))  # do not log raw document text
```

## SOLO exercise — Strengthen the ingestion boundary

Implement `validate_records(records)`. Accept an iterable of dictionaries with string `id`, `text`, and `label` fields. Missing required fields raise `ValueError`; wrong record/field types raise `TypeError`. Strip IDs and text, reject blanks or unknown labels with `ValueError`, and reject duplicate IDs after stripping. Return new dictionaries containing exactly those three fields. Preserve input records; empty input returns an empty list. Do not normalize label spelling silently.

```python exercise
def validate_records(records):
    raise NotImplementedError("Validate and canonicalize the ingestion records")

assert validate_records([{"id":" x ", "text":" claim ", "label":"claim"}]) == [{"id":"x", "text":"claim", "label":"claim"}]
```

**Completion gate:** run the whole workflow, report baseline and model metrics, reload the artifact without retraining, reject a malformed record, and route an unsupported request to review. Write a short handoff explaining the synthetic-data limitation, the feature contract, artifact lineage, restart path, and what should trigger investigation. Reference answer: `solutions/draft/ch26.py`; tests: `tests/draft/test_ch26.py`.

**Next enrichment:** replace the model with Study 09's pretrained adaptation, connect the request boundary to Study 16's server, and persist job status using Study 17. Compare the richer model on a representative evaluation set before promoting it. Primary references: [PyTorch learning workflow](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [saving/loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), and [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/).
