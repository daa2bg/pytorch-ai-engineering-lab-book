# Study 03 — Objects that feed a model

**Outcome:** build an indexable dataset that feeds aligned batches to PyTorch, with explicit preprocessing settings. This is the last short Python bridge before tensor work. The class examples use the standard library; the final loader requires the repository's PyTorch environment. Run worked cells in order and leave tagged exercise cells until your own attempt. Allow about 35 minutes including practice.

Our correspondence pipeline already transforms records into features and category IDs. Its next job is to supply the same samples reliably for every training epoch. If a source list changes midway through training, an index can refer to a different example; if preprocessing settings change, the same text can produce different features. Chapter 2 transformed one record into model inputs. A trainer also needs a consistent way to ask for sample 0, sample 1, and so on. A list can already do this. A small class becomes useful when fetching a sample also performs a transformation, or when configuration belongs with the data. We will use the interface that PyTorch's map-style datasets expect, without introducing inheritance yet.

## Keep an experiment's settings visible

A **configuration** records choices that affect an experiment. Consider lowercasing: changing that choice can change the model's inputs. An explicit configuration is easier to save and review than a global variable changed by another notebook cell.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TextConfig:
    lowercase: bool = True

config = TextConfig(lowercase=False)
assert config.lowercase is False
print(config)
```

`class` defines a type; `TextConfig(...)` constructs an instance. `@dataclass` is a decorator: it adds routine methods, including an initializer, from the declared fields. `frozen=True` prevents normal reassignment of fields after construction. This particular configuration contains only a Boolean, so there is no nested mutable state to share accidentally. A frozen dataclass containing a list would not make that list immutable.

The annotation `lowercase: bool` documents the expected type. It is not runtime type validation. Our small example assumes callers supply a Boolean. Larger ingestion boundaries can enforce richer schemas when necessary.

A configuration also needs an ordinary representation when we record a run. `asdict` exposes the fields; `**settings` unpacks a dictionary as keyword arguments when reconstructing the instance. Saving settings does not save a dataset or a trained model.

```python
from dataclasses import asdict
import json

config_text = json.dumps(asdict(config))
restored_config = TextConfig(**json.loads(config_text))
assert restored_config == config
print("Recorded preprocessing:", config_text)
```

## Implement the access contract

The next class owns a snapshot of the input texts and its configuration. `self` is the instance receiving a method call. Assigning `self.texts` makes that value available to later calls on the same object.

```python
class TextDataset:
    def __init__(self, texts, config=TextConfig()):
        self.texts = tuple(texts)
        self.config = config

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        if self.config.lowercase:
            text = text.lower()
        words = tuple(text.split())
        return words

source = ["Invoice DUE", "Claim received"]
dataset = TextDataset(source)
source.append("Added later")
assert len(dataset) == 2
assert dataset[0] == ("invoice", "due")
assert dataset[-1] == ("claim", "received")
print("First sample:", dataset[0])
```

Python calls `__len__` when we write `len(dataset)`, and `__getitem__` for `dataset[index]`. These double-underscore methods implement language operations; we do not normally call them directly. The tuple snapshot isolates the outer collection from later appends to `source`. The strings inside are immutable. The returned tuple of words is also stable for a caller to inspect.

Index behavior comes from the underlying tuple: an out-of-range integer raises `IndexError`, and negative indices count from the end. We have not invented a separate indexing scheme. A PyTorch `DataLoader` can request indices and group returned samples. The class need not inherit from a particular base class for this small map-style interface: the loader needs its length and indexed samples. Our token tuples demonstrate access; the numerical pair interface below supplies training inputs.

```python
try:
    dataset[2]
except IndexError:
    print("Out-of-range sample rejected")
else:
    raise AssertionError("Expected IndexError")
```

## Exercise — A dataset of feature/label pairs

Implement `PairDataset(features, labels)`. The caller supplies a sequence of numerical rows and a sequence of labels. Reject unequal lengths with `ValueError`. Snapshot each row as a tuple and the labels as a tuple, so later edits of source lists cannot change samples. Implement `__len__` and `__getitem__`; indexing returns `(feature_tuple, label)`. Let tuple indexing provide negative-index and `IndexError` behavior. Additional numeric-type validation is outside this exercise.

```python exercise
class PairDataset:
    def __init__(self, features, labels):
        raise NotImplementedError("Store aligned snapshots")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

rows = [[2.0, 5.0], [1.0, 3.0]]
pairs = PairDataset(rows, [0, 1])
rows[0][0] = 99.0
assert len(pairs) == 2
assert pairs[0] == ((2.0, 5.0), 0)
```

## Make the batch boundary explicit

A loader chooses samples; **collation** combines them into the representation the model expects. These are separate responsibilities. Our target is one numerical matrix with a row per sample and one aligned target vector. PyTorch's default collation recursively preserves nested structures, so a tuple of scalar features is not automatically treated as a feature vector. A short callback makes the conversion explicit.

The worked example uses ordinary pairs so it runs without your unfinished class. They have the same feature/label contract as `PairDataset`: two numeric features and a class ID from Chapter 2's synthetic `documents-demo.v1` example.

```python
import torch
from torch.utils.data import DataLoader

def collate_pairs(samples):
    rows = [features for features, label in samples]
    labels = [label for features, label in samples]
    return (torch.tensor(rows, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))

prepared_pairs = [((2.0, 5.0), 0), ((1.0, 3.0), 1), ((4.0, 2.0), 0)]
loader = DataLoader(prepared_pairs, batch_size=2, shuffle=False,
                    collate_fn=collate_pairs, num_workers=0)
for epoch in range(2):
    batch_sizes = []
    for features, targets in loader:
        assert features.shape == (len(targets), 2)
        batch_sizes.append(len(targets))
    assert batch_sizes == [2, 1]
print("Both epochs keep the final one-sample batch:", batch_sizes)
```

`collate_fn=collate_pairs` passes the function, not its result. The loader calls it with a list of samples. `float32` stores fractional features and `long` stores integer class IDs; Chapter 4 explains those tensor choices. `num_workers=0` keeps sample fetching in this process. We preserve order here to inspect alignment; Chapter 6 adds seeded shuffling for actual model training. Each `for ... in loader` obtains a new traversal, avoiding Chapter 1's exhausted-iterator bug.

Now replace `prepared_pairs` with your `PairDataset`. Verify that an edit to the original inputs does not alter either epoch's batch contents. **Hint:** snapshot each numeric row, not just the outer list. One tuple around the outer list would still share mutable child rows. The caller promises numeric entries and scalar labels; this snapshot is not a general deep-copy facility.

```python exercise
source_rows = [[2.0, 5.0], [1.0, 3.0], [4.0, 2.0]]
source_labels = [0, 1, 0]
owned_pairs = PairDataset(source_rows, source_labels)
source_rows[0][0] = 999.0
source_labels[0] = 1
owned_loader = DataLoader(owned_pairs, batch_size=2, shuffle=False,
                          collate_fn=collate_pairs, num_workers=0)
for epoch in range(2):
    rows_seen, labels_seen = [], []
    for features, targets in owned_loader:
        rows_seen.extend(features.tolist())
        labels_seen.extend(targets.tolist())
    assert rows_seen == [[2.0, 5.0], [1.0, 3.0], [4.0, 2.0]]
    assert labels_seen == [0, 1, 0]
```

**Optional study:** why might a dataset that reads images from disk store paths rather than eagerly loading every image? Distinguish the reusable dataset object from a one-shot iterator over it.

**Completion gate:** retrieve a pair, demonstrate source isolation through two actual loader epochs, retain the final batch, and test unequal lengths and an out-of-range index. Explain which setting changes tokenization and which function assembles the tensor batch. You have enough class machinery to understand the small datasets used by a first trainer. Next we make the numerical rows into tensors; additional Python protocols can wait until an implementation needs them.

References: [Python classes](https://docs.python.org/3.11/tutorial/classes.html), [dataclasses](https://docs.python.org/3.11/library/dataclasses.html), [PyTorch datasets](https://docs.pytorch.org/docs/2.8/data.html). Separate answer: `solutions/draft/ch03.py`; tests: `tests/draft/test_ch03.py`.
