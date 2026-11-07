# Chapter 1 — Following the Data: From Everyday Loops to Machine Learning Pipelines

> **You do not need prior ML experience for this chapter.** We will build a small, realistic document-processing application from the ground up. Along the way, you'll learn what iteration is, how Python loops work, why generators exist, and how these ideas lead naturally to PyTorch's data-loading machinery.

**Estimated study time:** 2–3 hours including independent exercises. **Prerequisites:** variables, lists, functions, basic `for` loops. Every other Python concept is introduced here.

## The application we'll build

Imagine a company receives scanned letters, email attachments, and forms. A document-processing system needs to decide whether each item is a *claim*, an *invoice*, or *general correspondence*. A human has labeled some historical documents. We want to train a small classifier that learns patterns from those examples.

We won't train a neural network yet. First we need to deliver clean examples to the training code. That is a real engineering problem in its own right.

A **sample** is one example: perhaps a document's text together with the correct category. A **dataset** is a collection of samples. A **feature** is information the model uses (for example, word counts or a numerical text representation). A **label** is the known answer used in supervised learning. A **model** maps features to predictions; **training** adjusts the model's parameters so its predictions better match labels.

A typical supervised learning workflow looks like this:

1. Read a sample from storage.
2. Validate and transform it into usable features.
3. Collect several samples into a **batch**.
4. Pass the batch through a model.
5. Compare predictions with labels to calculate a **loss**.
6. Update model parameters to reduce the loss.
7. Repeat across the dataset, often many times.

A full pass over training data is called an **epoch**. The number of samples processed together is the **batch size**. Later, PyTorch will handle much of this plumbing, but understanding it will help you debug real training jobs.

**Connections ahead:** dataset streaming → generators; batching → `DataLoader`; repeated epochs → iterator exhaustion; validation → data quality; shuffling → learning dynamics; memory usage → scalable pipelines.

Official ML introduction: https://docs.pytorch.org/tutorials/beginner/basics/intro.html

```python
documents = [
    {"id": "doc-001", "text": "Please reimburse my hospital bill", "label": "claim"},
    {"id": "doc-002", "text": "Invoice number 4021 is due", "label": "invoice"},
    {"id": "doc-003", "text": "My mailing address has changed", "label": "general"},
    {"id": "doc-004", "text": "Claim for vehicle repairs", "label": "claim"},
    {"id": "doc-005", "text": "Payment reminder for invoice", "label": "invoice"},
]
assert len(documents) == 5
print("One sample:", documents[0])
print("Dataset size:", len(documents))
```

## 1. Start with the simplest possible loop

Suppose we want to inspect all documents and count their labels. We already know how to do that with a `for` loop. Here, *iteration* simply means visiting one item after another.

This first implementation is completely reasonable when data fits in memory. Do not reach for a generator merely because the project uses AI.

```python
counts = {}
for document in documents:
    label = document["label"]
    counts[label] = counts.get(label, 0) + 1
assert counts == {"claim": 2, "invoice": 2, "general": 1}
print(counts)
```

### What is happening under the hood?

The list is **iterable**: Python knows how to obtain an iterator from it. The iterator is an object that remembers where it is. Each `next()` call returns another sample; after the last sample it raises `StopIteration`. A `for` loop handles that exception automatically.

Think of a book and a bookmark. The book (iterable) can be read again from the beginning. The bookmark (iterator) represents one reading session and moves forward. A new bookmark can start at page one, but an exhausted bookmark cannot magically restart.

In Python terminology:
- `iter(container)` creates an iterator.
- `next(iterator)` requests the next item.
- `StopIteration` signals that there are no more items.

**Predict:** will `first` and `second` be the same object? Will they return the same first document?

```python
first = iter(documents)
second = iter(documents)
assert first is not second
assert next(first)["id"] == "doc-001"
assert next(first)["id"] == "doc-002"
assert next(second)["id"] == "doc-001"
print("Two independent reading positions.")
```

## 2. Why not always use a list?

For five documents, lists are easy and appropriate. For millions of documents, loading every file, decoding it, and storing its full text in memory can become expensive. Even if the metadata fits in memory, the raw images might not.

A useful approach is to read and transform **one document at a time**. Python supports this with iterators. A plain list iterator already visits one stored item at a time, but the list itself has been loaded eagerly. To delay *loading or computing* items until requested, we can write a generator.

**Key distinction:** an iterator is an interface for stepping through values; a generator is a convenient way to implement an iterator using `yield`. A generator does not automatically save memory if you already constructed a giant list before passing it in.

Compare `return` and `yield`: `return` finishes a function and returns a value; `yield` produces one value and suspends the function, preserving its local state until the next request.

```python
def generate_documents(records):
    for record in records:
        print("Loading", record["id"])
        yield record

stream = generate_documents(documents)
print("Generator created; no records loaded yet.")
first_record = next(stream)
assert first_record["id"] == "doc-001"
print("Now one record has been requested.")
```

### Generator lifecycle and exhaustion

Calling a generator function returns a generator object; it does **not** immediately execute the function body. Each request resumes execution until the next `yield`. Once exhausted, the same generator stays exhausted.

For another epoch, call the generator **function again** to create a fresh generator. A `DataLoader` may be reusable across epochs, but a one-shot iterator created from a generator is not.

**Real failure scenario:** a training script creates `stream = generate_documents(...)` outside its epoch loop. Epoch 1 processes all samples. Epoch 2 sees no samples. The code may appear to finish successfully without learning anything further.

```python
def numbers():
    yield 10
    yield 20

stream = numbers()
assert list(stream) == [10, 20]
assert list(stream) == []
assert list(numbers()) == [10, 20]
for epoch in range(2):
    print("Epoch", epoch, "items", list(numbers()))
```

## 3. Ordinary iterator class versus generator function

Both approaches implement the same iterator protocol. A class-based iterator is useful when you want explicit state and methods; a generator is usually shorter for straightforward streaming transformations.

The class below manually stores its current position. Its `__iter__` returns itself; its `__next__` advances or raises `StopIteration`. The generator version below it has the same observable output, but Python manages the position and suspension automatically.

Neither is inherently faster in all situations. Prefer readability first, then benchmark real workloads.

```python
class DocumentIterator:
    def __init__(self, records):
        self.records = records
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.records):
            raise StopIteration
        result = self.records[self.position]
        self.position += 1
        return result

def document_generator(records):
    for record in records:
        yield record

assert [d["id"] for d in DocumentIterator(documents)] == [
    d["id"] for d in document_generator(documents)
]
print("Class-based iterator and generator return the same sequence.")
```

### Three different things that are easy to confuse

1. **List:** stores all its items; iterating it normally creates a fresh list iterator.
2. **Iterator object:** remembers one traversal position; often exhausted after one pass.
3. **Generator function:** creates a generator iterator whenever called. The generator object itself is one-shot.

**Connected Python topic:** a *generator expression* uses parentheses: `(transform(x) for x in data)`. A list comprehension uses brackets: `[transform(x) for x in data]`. The generator expression defers computation, whereas the list comprehension constructs its output list immediately.

See: https://docs.python.org/3/tutorial/classes.html#iterators and https://docs.python.org/3/tutorial/classes.html#generators

```python
processed_list = [record["text"].lower() for record in documents]
processed_generator = (record["text"].lower() for record in documents)
assert len(processed_list) == 5
assert next(processed_generator) == processed_list[0]
assert len(list(processed_generator)) == 4
print("A list is reusable; this generator expression was consumed.")
```

## 4. Turning a document stream into a processing pipeline

A realistic application rarely reads a sample and immediately sends it to a model. We validate, normalize and transform it first. The pipeline below uses two small generators: one reads documents, another transforms them.

This separation makes each stage testable and replaceable. It also means that data can flow through the pipeline without first collecting all transformed documents in a new list.

**Important:** the example operates on in-memory records to keep the lesson offline and reproducible. In production, the first stage could read database pages, a file iterator, or object-storage responses.

```python
def normalized_documents(records):
    for record in records:
        text = record["text"].strip().lower()
        if not text:
            continue
        yield {"id": record["id"], "tokens": text.split(), "label": record["label"]}

pipeline = normalized_documents(document_generator(documents))
first_sample = next(pipeline)
assert first_sample["tokens"] == ["please", "reimburse", "my", "hospital", "bill"]
print(first_sample)
```

## 5. Why batches exist

A model can process one example at a time, but processing many together often uses vectorized numerical operations and accelerators more efficiently. Instead of making five separate model calls, we can group samples into batches.

A batch is not an epoch. With five samples and batch size two, one epoch consists of three batches of sizes 2, 2 and 1. Some training setups discard the final incomplete batch; others keep it.

Batching is also where data from different samples must be combined into compatible shapes. Later, a text model may need to **pad** variable-length token sequences and create an **attention mask** so padded positions are ignored. That is why batching is more than a Python list operation.

```python
def reference_batches(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

all_batches = list(reference_batches(normalized_documents(documents), 2))
assert [len(b) for b in all_batches] == [2, 2, 1]
print("Batch sizes:", [len(b) for b in all_batches])
```

## 6. What PyTorch will do later

PyTorch provides `Dataset` to describe how examples are accessed and `DataLoader` to assemble them into batches. A map-style dataset usually supports indexing; an `IterableDataset` can represent streams. `DataLoader` can handle batching, shuffling for supported datasets, and parallel data loading.

This chapter does not require PyTorch. The point is to understand the concepts first, then recognize them in a framework rather than memorizing an API without context.

**Illustrative future code (not executed in this chapter):**

```text
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for features, labels in loader:
    predictions = model(features)
    loss = loss_function(predictions, labels)
    # backward, optimizer update, repeat
```

Here `dataset`, `model`, `loss_function`, and the optimizer must be created separately. This is pseudocode for the training structure, not a complete runnable example.

Official walkthrough: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

## 7. SOLO lab: implement your own streaming batcher

Now write `my_batches(items, size)` using `yield`. It must:
- Accept a list **or a one-shot iterator**.
- Preserve order and yield lists containing at most `size` samples.
- Yield a smaller final batch instead of silently losing examples.
- Produce no batches for empty input.
- Raise `ValueError` for zero or negative size.
- Avoid materializing all input items at once.

**Before coding:** write a few lines describing your algorithm in plain English. Then implement it without Copilot. Finally design tests before viewing the reference solution.

**Python subtlety:** code inside a generator function is executed when iteration starts, not at the moment the function is called. Your validation may therefore raise on `list(my_batches(...))` rather than on `my_batches(...)`.

```python exercise
def my_batches(items, size):
    # SOLO: implement here. Replace the placeholder.
    raise NotImplementedError

assert list(my_batches(iter(range(5)), 2)) == [[0, 1], [2, 3], [4]]
```

## 8. SOLO lab: reproduce and fix an epoch bug

A developer writes (illustrative broken loop; `train_on` is a placeholder):

```text
stream = normalized_documents(documents)
for epoch in range(3):
    for sample in stream:
        train_on(sample)
```

Without running it, explain which epoch processes data. Then implement a corrected version using a fresh generator for each epoch. Add a counter asserting that each epoch processes five samples.

**Stretch:** suppose your data source is a non-rewindable network stream. What must change to train for multiple epochs? Options include reopening the source, caching it, or using a different training strategy. Explain the memory and reliability tradeoffs.

Implement `count_epochs(records, epochs)` for a reusable list of records. Return the number of nonblank-text records seen in each epoch. Create a fresh `normalized_documents(records)` stream inside the epoch loop. Return an empty list for zero epochs and raise `ValueError` for negative epochs. The caller supplies an integer epoch count and valid record fields.

```python exercise
def count_epochs(records, epochs):
    raise NotImplementedError("Create a fresh normalized stream for every epoch")

assert count_epochs(documents, 3) == [5, 5, 5]
assert count_epochs(documents, 0) == []
```

## 9. SOLO lab: read a realistic ingestion source

Our document stream currently begins with a list. A JSONL file stores one JSON value per line, so we can read a record without loading the whole file. `json.loads(line)` converts one line of JSON text to a Python value; it can raise `json.JSONDecodeError` when the text is malformed. A valid JSON value is not necessarily an object: `[]` is valid JSON but is not a document record.

Implement `read_documents(path)` as a generator. Use `with open(path, encoding="utf-8")` and visit physical lines with `enumerate(source, 1)`. Follow this small, explicit policy:

- Skip blank lines.
- Require each remaining line to be a JSON object containing `id`, `text`, and `label`; preserve extra fields. Detailed field types and label rules are developed in the data-contract study.
- Raise `ValueError` with the physical line number for malformed JSON, a non-object value, or missing required keys.
- Let file errors and invalid UTF-8 propagate; do not silently skip corrupted input.
- Keep the source open only while the stream is active. Exhaustion or a raised error exits the `with` block. A consumer stopping early must call the generator's `close()` method, preferably in its own `finally` block.

**Hint:** parse inside `try`/`except json.JSONDecodeError`, validate the resulting value, then `yield` the record. This is a single-record transformation repeated over the file; do not call `readlines()` or convert the file into a list.

**Why this matters:** an input source that hides malformed records can distort the data a model learns from. A stream suspended at a `yield` also still owns its open file, so early stopping needs an explicit cleanup policy.

**Optional challenge:** design a resume policy using stable record IDs. Explain which failures can replay records and how a later stage could avoid duplicated work. This is not required for the first streaming reader.

References: [Python JSON decoding](https://docs.python.org/3/library/json.html), [reading files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files), and [generator methods](https://docs.python.org/3/reference/expressions.html#generator-iterator-methods).

```python exercise
def read_documents(path):
    raise NotImplementedError("Read, validate, and yield one JSON object per line")

import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temporary:
    source_path = Path(temporary) / "documents.jsonl"
    source_path.write_text('\n{"id":"a","text":"hello","label":"general"}\n', encoding="utf-8")
    assert list(read_documents(source_path)) == [
        {"id": "a", "text": "hello", "label": "general"}
    ]
```

## 10. Self-check: understand before you proceed

Explain in your own words:

1. What is the difference between a sample, dataset, batch and epoch?
2. Why is a list iterable but a generator object normally one-shot?
3. How do `iter()`, `next()`, `__iter__()` and `__next__()` relate?
4. What does `yield` preserve between calls?
5. Does replacing a list with a generator always save memory?
6. Why might training use batches instead of individual samples?
7. What can go wrong if you use the same generator for three epochs?
8. Why can a validation error in a generator occur later than expected?
9. How would you test a stream that raises an error midway through?
10. When should you prefer a plain list for clarity?

**Completion gate:** finish the three SOLO labs; write tests for empty input, incomplete batches and bad batch size; demonstrate the epoch fix; explain all ten questions.

**Further reading (optional, not a prerequisite):**
- Python iterator protocol and generators: https://docs.python.org/3/tutorial/classes.html#iterators
- Python functional programming HOWTO: https://docs.python.org/3/howto/functional.html
- PyTorch datasets and data loaders: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch iterable datasets: https://docs.pytorch.org/docs/stable/data.html

**Next chapter:** functions, mutability and ownership. We'll ask a new practical question: *What happens if preprocessing one sample accidentally modifies the original dataset?*

Separate answers: `solutions/book-01/ch01_streams.py`. Exercise tests: `tests/test_ch01_streams.py`. Run `PYTHONPATH=src python -m pytest tests/test_ch01_streams.py` (in PowerShell, set `$env:PYTHONPATH="src"` first).
