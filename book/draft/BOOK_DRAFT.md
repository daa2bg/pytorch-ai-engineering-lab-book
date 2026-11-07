# PyTorch & AI Engineering Lab — First Draft

A practical path from experienced software developer to ML/AI engineering. This is a working first draft for study and refinement, not a release candidate. Run each chapter's own notebook; solutions and tests are separate. See VALIDATION.md for actual execution evidence and skipped integrations.

## Reading map

- 01. Following the data: iterators and batching (core)
- 02. From Python functions to model inputs (core)
- 03. Objects that feed a model (core)
- 04. Tensors: shape is part of the interface (core)
- 05. Autograd: make one parameter learn (core)
- 06. Train your first PyTorch classifier (core)
- 07. Measure, split, and reproduce (core)
- 08. Use a pretrained tokenizer and model (core)
- 09. Fine-tune a small pretrained model (core)
- 10. Turn incoming files into a usable dataset (core)
- 11. Join and inspect data before training (core)
- 12. Version the data transformation (core)
- 13. Search by vectors and evaluate the neighbors (core)
- 14. Build an evidence-first RAG workflow (core)
- 15. Run inference with measurable tradeoffs (core)
- 16. Expose the model through a small service (core)
- 17. Compose restartable AI workflow steps (core)
- 18. Make failures visible without leaking data (core)
- 19. Match architecture to input structure (study)
- 20. See what attention computes (study)
- 21. Train a tiny language-model experiment (study)
- 22. Adapt fewer parameters with LoRA (study)
- 23. Study predictive error and replay (study)
- 24. Keep provider integrations replaceable (study)
- 25. Learn orchestration by producing this book (study)
- 26. Deliver an end-to-end ML/AI workflow (core)

---

## Study 01 — Following the data: iterators and batching

Protected existing reference. Source notebook: `book/01-python-for-ai/01-iterators-and-batching.ipynb`.

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

---

## Study 02 — From Python functions to model inputs

Protected existing reference. Source notebook: `book/01-python-for-ai/02-functions-and-data-model.ipynb`.

# Chapter 2 — From Python Functions to Model Inputs

Chapter 1 showed how to move samples through a stream and collect them into batches. Now we need to decide what each sample contains. A training loop cannot do much with a dictionary holding raw text and a category name: it needs consistent numerical inputs and targets.

Our task is to build that preparation step with ordinary Python. Along the way, we will use functions, keyword arguments, dictionaries, comprehensions, and functions passed as arguments. We will also catch two Python mistakes that can silently change an experiment. At the end, we will convert the prepared data into PyTorch tensors.

**Prerequisites:** basic variables, lists, dictionaries, loops, and Chapter 1's iterable/batch ideas. Allow approximately 60–90 minutes including three short exercises. Core examples use the standard library; the final tensor bridge runs if PyTorch is installed. All examples and prompts are together in the companion notebook. Run worked cells in order and skip tagged unfinished exercise cells until you attempt them. Reference answers and tests are separate.

## 1. A small dataset and a clear transformation

A **sample** is one example. A **label** is its known answer; in supervised learning, training uses these answers to improve a model's predictions. A **feature** is information supplied to the model. For now, our features will be simple text measurements, rather than the learned text representations used by larger models.

```python
documents = [
    {"id": "d1", "text": " Invoice   payment due ", "label": "invoice"},
    {"id": "d2", "text": "Please review my claim", "label": "claim"},
    {"id": "d3", "text": "Invoice received", "label": "invoice"},
]
label_to_id = {"claim": 0, "invoice": 1}

assert len(documents) == 3
print(documents[0])
```

A dictionary connects names to values: `record["text"]` selects a field. Our output should preserve the document ID, replace text with a small list of numbers, and represent the label using an integer. The ID helps us trace a bad result back to its source; it is not a model feature.

The exact measurements are intentionally simple. This chapter teaches the Python mechanisms used to prepare model inputs. A useful classifier will need representative data, a suitable representation, training, and evaluation.

## 2. Functions make preprocessing reusable

Before extracting numbers, we need a repeatable way to split text. For this small example, a **token** is one whitespace-separated piece. Model tokenizers may split text differently, including into parts of words. We are not reproducing a pretrained tokenizer here.

```python
def tokenize(text, *, lowercase=True):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if lowercase:
        text = text.lower()
    return text.split()

assert tokenize(" Invoice   Due ") == ["invoice", "due"]
assert tokenize("Invoice Due", lowercase=False) == ["Invoice", "Due"]
assert tokenize("   ") == []
print(tokenize(documents[0]["text"]))
```

`def` defines the function; its indented body runs when called. `text` and `lowercase` are parameters, while the values supplied at the call are arguments. `return` hands the result to the caller. Printing a value displays it but does not return it; a function with no return value returns `None`.

The bare `*` makes `lowercase` keyword-only: use `lowercase=False`, which explains the setting at the call site. `True` is the default when omitted. String methods return new strings, so rebinding the local name `text` does not edit the caller's original string. `split()` without a separator handles repeated whitespace and omits empty pieces.

Notice the type check. A wrong object should fail here with a clear error, rather than much later in a training run. We use `TypeError` for the wrong kind of input. We do not reject blank text: its empty-token behavior is part of this function's contract.

```python
try:
    tokenize(17)
except TypeError as error:
    print("Rejected:", error)
else:
    raise AssertionError("Expected TypeError")

original_text = " Invoice   Due "
assert tokenize(original_text) == ["invoice", "due"]
assert original_text == " Invoice   Due "
```

`try` runs a potentially failing operation; `except` handles the specified exception. The `else` branch runs only when no exception occurred, so this example detects a missing rejection. Assertions are useful executable expectations, but explicit exceptions should enforce public input rules: Python can disable assertions under optimization.

**Try changing the policy:** preserve case, then compare whether `"Invoice"` matches the lowercase word `"invoice"`. Preprocessing settings affect the features a model sees. Training and inference must use compatible settings.

## 3. Turn text into a feature vector

A **feature vector** is an ordered collection of numerical measurements for one sample. The order is part of the contract. If the first number means word count during training, it must mean word count during prediction too.

Start with two easily checked measurements: number of words and whether the exact word `invoice` occurs.

```python
def demo_features(text):
    words = tokenize(text)
    return [float(len(words)), float("invoice" in words)]

assert demo_features("Invoice payment due") == [3.0, 1.0]
assert demo_features("Please review my claim") == [4.0, 0.0]
assert demo_features("invoices") == [1.0, 0.0]
print("Feature order: [word_count, contains_invoice]")
```

`len` counts the words; `in` tests membership. A Boolean converts to `1.0` or `0.0` with `float`. Explicit floats will fit the numerical inputs of the small PyTorch model we will build later. Substring matching would be a different feature: our membership test does not count `invoices` as `invoice`.

For a small batch, a list comprehension expresses “apply this function to each item.” The loop version is equally valid when you need more debugging steps.

```python
feature_rows = [demo_features(record["text"]) for record in documents]
assert feature_rows == [[3.0, 1.0], [4.0, 0.0], [2.0, 1.0]]
print(feature_rows)
```

### Exercise 1 — Implement another useful pair of features

Write `text_features(text)` returning `[word_count, mean_word_length]`, both floats. Split on whitespace, preserve punctuation inside each word, and return `[0.0, 0.0]` for blank text. Reject non-string input with `TypeError`. Do not modify the input.

The mean is the sum of word lengths divided by the number of words. Measure after lowercasing, as `tokenize` does by default; Unicode case conversion can sometimes change a word's length. Start with a plain loop that adds `len(word)`; only divide when there is at least one word. You may reuse `tokenize` in the learner implementation. This is a tiny feature extractor, not a complete text representation.

```python exercise
def text_features(text):
    # SOLO: return two floats; handle blank text before division.
    raise NotImplementedError("Exercise 1")

assert text_features("Small model") == [2.0, 5.0]
assert text_features("  ") == [0.0, 0.0]
assert text_features("Hi!") == [1.0, 3.0]
```

## 4. Make transformation choices explicit

We can use the same preparation function with different feature extractors. Python functions are objects: we can assign them to names and pass them as arguments. Pass `demo_features` to supply the function; write `demo_features(text)` to call it immediately.

This is the foundation of a **callback**: a function another piece of code calls to perform a chosen step. Dataset transformations and batch-building hooks use this idea.

```python
def encode_record(record, label_to_id, *, feature_fn=demo_features):
    return {
        "id": record["id"],
        "features": feature_fn(record["text"]),
        "label_id": label_to_id[record["label"]],
    }

encoded = encode_record(documents[0], label_to_id)
assert encoded == {"id": "d1", "features": [3.0, 1.0], "label_id": 1}
print(encoded)

def character_features(text):
    return [float(len(text))]

assert encode_record(documents[0], label_to_id,
                     feature_fn=character_features)["features"] == [23.0]
```

The `feature_fn` default is a function object. Inside `encode_record`, we call it with the text. The alternative character-count feature illustrates replacing behavior without rewriting the record-handling code. These two extractors have different output widths; never mix them within the same model input batch.

The label dictionary provides a stable **encoding**: category name to integer ID. These IDs identify categories; their magnitude is not a ranking. Preserve the mapping with the model to interpret predictions and encode any known labels used in evaluation. Do not independently rebuild it from each batch: a different order could make the same ID mean a different class. Label lookup is exact: `" Invoice "` is not the same key as `"invoice"`. Do not pass the target label into a feature extractor; that would give a model the answer it is supposed to predict.

Missing fields and unknown labels currently raise `KeyError` through dictionary lookup. This helper assumes records have the documented fields and a valid category mapping. At an ingestion boundary, add schema validation once; do not obscure defects by returning fabricated empty records.

### Scope and type hints, in the amount we need

A function can read global names, but mutable global settings can hide why two calls behave differently. Pass experiment settings explicitly, as we do with `label_to_id` and `feature_fn`. Local assignments do not replace a caller's variable; mutation of a shared object can still affect the caller.

Type hints help describe contracts to an editor or checker. For example, `def label_id(label: str, mapping: dict[str, int]) -> int:` describes a string argument, a string-to-integer dictionary, and an integer return. Ordinary Python does not enforce those hints at runtime. Add them when useful; a small function does not need an elaborate typing framework to be testable.

## 5. Preserve raw samples and avoid hidden state

Feature extraction should leave source data available for another experiment. Assignment does not copy a dictionary: two names can refer to the same object. The `is` operator tests identity, whereas `==` compares values.

```python
raw = {"text": "Invoice Due", "tags": ["incoming"]}
alias = raw
assert alias is raw
preview = raw.copy()
preview["text"] = preview["text"].lower()
assert raw["text"] == "Invoice Due"
assert preview is not raw
assert preview["tags"] is raw["tags"]
print("New outer dictionary; nested tags still shared.")
```

A dictionary's `copy()` is shallow: it copies the outer container while keeping references to its values. Changing `preview["text"]` replaces a slot only in the copied dictionary. Appending to its shared tags list would change the source's tags too. Copy that child if your transformation needs to edit it. Our encoded samples contain freshly created feature lists and deliberately omit raw metadata; no deep copy of the dataset is needed.

There is another easy way to share a list accidentally: a mutable default. Defaults are evaluated when the `def` statement runs, not once per call.

```python
def collect_bad(value, output=[]):
    output.append(value)
    return output

first = collect_bad("batch-1")
collect_bad("batch-2")
assert first == ["batch-1", "batch-2"]

def collect_fresh(value, output=None):
    if output is None:
        output = []
    output.append(value)
    return output

assert collect_fresh("batch-1") == ["batch-1"]
assert collect_fresh("batch-2") == ["batch-2"]
print("Omitted output now starts fresh on each call.")
```

The `None` marker lets the function allocate a fresh list inside the call. If a caller supplies a list, this version deliberately edits it. The broader lesson is to know which state belongs to a sample, a batch, or the whole run. A collector reused unexpectedly across batches can duplicate data and make results depend on call order.

### Exercise 2 — Prepare a sample for your model

Write `prepare_sample(record, label_to_id)`. Use your `text_features` from Exercise 1 and return a **new dictionary** with exactly `id`, `features`, and `label_id`. Preserve the original record. Required input fields are `id`, `text`, and `label`; the supplied mapping associates category strings with integer IDs. Missing fields or an unknown label raise `KeyError`; non-string text raises `TypeError`. No additional ID or mapping validation is required.

```python exercise
def prepare_sample(record, label_to_id):
    # SOLO: assemble a new record using your feature function.
    raise NotImplementedError("Exercise 2")

sample = {"id": "x", "text": "Small model", "label": "claim"}
prepared = prepare_sample(sample, {"claim": 0})
assert prepared == {"id": "x", "features": [2.0, 5.0], "label_id": 0}
assert prepared is not sample
assert sample == {"id": "x", "text": "Small model", "label": "claim"}
```

## 6. Assemble the batch: features and targets stay aligned

A batch groups samples for processing together. Model inputs and their correct answers must preserve the same order. Here is the ordinary loop version, using our worked extractor so it does not depend on your exercise solutions.

```python
inputs = []
targets = []
for record in documents:
    item = encode_record(record, label_to_id)
    inputs.append(item["features"])
    targets.append(item["label_id"])

assert inputs == [[3.0, 1.0], [4.0, 0.0], [2.0, 1.0]]
assert targets == [1, 0, 1]
assert len(inputs) == len(targets)
print("Inputs:", inputs, "Targets:", targets)
```

The outer input list has one row per sample; every inner list has two features in the same order. The target at position zero describes input row zero. Separate filtering or sorting of these lists would break that pairing.

Chapter 1's streaming batcher can pass one small list of records into this step at a time. A Python list still stores this batch in memory, which is appropriate: streaming a dataset does not require eliminating every small intermediate list.

### Exercise 3 — Connect a stream to a batch builder

Write `make_batch(records, label_to_id)` using `prepare_sample`. Accept a list or a one-shot iterator, traverse it once, and return `(feature_rows, label_ids)`. This is a tuple containing two lists; the comma in a return expression packs the pair, and `x, y = ...` unpacks it.

Preserve order, do not modify source records, and return `([], [])` for empty input. Let earlier errors propagate. An iterator is consumed, and it may be partly consumed before a bad record raises an error; no rollback is promised.

```python exercise
def make_batch(records, label_to_id):
    # SOLO: build two aligned lists in one traversal.
    raise NotImplementedError("Exercise 3")

records = [
    {"id": "a", "text": "Small model", "label": "claim"},
    {"id": "b", "text": "Hi!", "label": "invoice"},
]
x, y = make_batch(iter(records), {"claim": 0, "invoice": 1})
assert x == [[2.0, 5.0], [1.0, 3.0]]
assert y == [0, 1]
assert make_batch([], {"claim": 0}) == ([], [])
```

**Optional study:** add a keyword-only `feature_fn` parameter to the exercise functions. Test a one-feature extractor and a two-feature extractor separately. Decide where to reject inconsistent row widths before tensor construction. Keep this extension outside the essential completion gate.

## 7. The next boundary is a PyTorch tensor

A **tensor** is a numerical array used by PyTorch. A **shape** describes its dimensions; a **dtype** describes the stored number type. Our inputs should have shape `[batch_size, feature_count]`; integer targets should have shape `[batch_size]`.

The following is a real executable bridge, not a training loop. It uses the worked `inputs` and `targets`, so it runs before you finish the exercises. If PyTorch is absent, it prints an explicit skip; use the repository's configured Python environment when you want to run it.

```python
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    print("SKIP tensor bridge: PyTorch is not installed in this environment.")
else:
    x_tensor = torch.tensor(inputs, dtype=torch.float32)
    y_tensor = torch.tensor(targets, dtype=torch.long)
    assert tuple(x_tensor.shape) == (3, 2)
    assert tuple(y_tensor.shape) == (3,)
    assert y_tensor.tolist() == [1, 0, 1]
    print("PyTorch inputs:", x_tensor.shape, x_tensor.dtype)
    print("PyTorch targets:", y_tensor.shape, y_tensor.dtype)
```

Floating-point features and integer class IDs are a common pairing for a classifier trained with PyTorch's cross-entropy loss, which measures prediction error against the class targets. Other tasks and loss functions have different target contracts. No parameters have been learned in this chapter: tensor conversion only prepares numbers for computation.

Next, a small model will turn each feature row into class scores. A training loop will compare those scores with targets and adjust the model's parameters to reduce error. Fine-tuning uses the same basic cycle while starting from pretrained parameters. The feature function may become a pretrained tokenizer, but the contracts and alignment you practiced here still matter.

**Early applied route:** finish the compact Python bridge, learn tensor operations and the minimum autograd concepts, train and evaluate a small model, then use a pretrained Hugging Face model and perform a small fine-tuning run. You do not need to build a transformer or language model from scratch before using and fine-tuning one. The curriculum keeps those deeper studies available afterward.

## Completion check

Before moving on, implement the three exercises without copying the reference answers. Demonstrate a normal sample, blank text, an unknown label, an unchanged source record, and a batch built from an iterator. Explain why feature order and label IDs must stay consistent between training and inference.

Restart the kernel and execute all worked cells while skipping unfinished exercise cells. If PyTorch is installed, confirm the two tensor shapes. This chapter's deliverable is a tested sample-to-batch transformation, not a general validation framework or a trained model.

### References and files

- Python functions, defaults, and keyword-only parameters: https://docs.python.org/3.11/tutorial/controlflow.html#defining-functions
- Python containers and comprehensions: https://docs.python.org/3.11/tutorial/datastructures.html
- Copying and shared values: https://docs.python.org/3.11/library/copy.html
- Type hints and their runtime limits: https://docs.python.org/3.11/library/typing.html
- PyTorch tensor fundamentals: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
- PyTorch dataset transformations: https://docs.pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
- Hugging Face fine-tuning overview (for the early applied route): https://huggingface.co/docs/transformers/training

Complete learner notebook: `book/01-python-for-ai/02-functions-and-data-model.ipynb`. Reference answers: `solutions/book-01/ch02_functions.py`. Tests: `tests/test_ch02_functions.py`. Build with `python tools/build_ch02_notebook.py`; check synchronization with `python tools/build_ch02_notebook.py --check`. Fences labeled `python exercise` become tagged, intentionally unfinished learner cells.

---

## Study 03 — Objects that feed a model

New first-draft study. Source notebook: `book/01-python-for-ai/03-classes-protocols-and-configuration.ipynb`.

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

---

## Study 04 — Tensors: shape is part of the interface

New first-draft study. Source notebook: `book/draft/04-tensor-shapes.ipynb`.

# Study 04 — Tensors: shape is part of the interface

**Outcome:** turn feature rows into tensors, fit a reusable normalization on training rows, and diagnose shape and leakage errors. PyTorch is required; no GPU or download is needed. We use small synthetic document measurements so the arithmetic is visible before the real digits experiment in Chapter 6. Allow about 40 minutes including practice.

Our correspondence pipeline now delivers aligned feature/label pairs. Its next boundary is numerical: the same feature must retain the same meaning when a training batch becomes a later prediction request. A Python list can hold our feature rows, but a model needs numerical operations over whole batches. A **tensor** is PyTorch's numerical array. Its **shape** states how many values lie along each dimension, its **dtype** states their number representation, and its **device** states where they live. These are part of the interface between preprocessing and the model, not incidental debugging details.

## Name the dimensions before calculating

We have three documents and two features per document: word count and an illustrative score. Dimension 0 selects samples; dimension 1 selects features. An **axis** is another name for a dimension.

```python
import torch

x = torch.tensor([[2, 10], [4, 10], [6, 10]], dtype=torch.float32)
assert x.shape == (3, 2)
assert x.device.type == "cpu"
assert x.dtype == torch.float32
print("Samples, features:", tuple(x.shape))
print("One sample:", x[0].tolist())
```

Explicit floating-point storage matters: a count may start as an integer, while an average or gradient needs fractional values. Classification targets often use integer class IDs instead. Do not cast everything to the same type merely because it belongs to one batch.

## Let a column statistic describe the operation

Different feature scales can make optimization awkward. A common preparation step subtracts each training column's mean and divides by its standard deviation. The mean tells us the center; the standard deviation measures spread. A constant column has no spread, so we use a scale of one for that column rather than divide by zero.

```python
mean = x.mean(dim=0, keepdim=True)
spread = x.std(dim=0, correction=0, keepdim=True)
scale = torch.where(spread == 0, torch.ones_like(spread), spread)
normalized = (x - mean) / scale

assert mean.shape == (1, 2)
assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
assert torch.equal(normalized[:, 1], torch.zeros(3))
print("Column means:", mean.tolist())
print("Normalized:", normalized.tolist())
```

`dim=0` reduces across samples, producing one statistic per feature. `keepdim=True` retains a length-one sample dimension. **Broadcasting** allows that `[1, 2]` statistic to apply to each row of the `[3, 2]` input. We did not make three copies manually. `correction=0` uses the population standard deviation for this finite training table; it also avoids an undefined corrected estimate when there is only one row.

The statistics must be learned from training data only, then reused for validation and inference. **Fitting** estimates settings from examples; **transforming** applies those settings. Mixing these operations lets held-out information influence development, a form of data leakage. It also makes a prediction depend on what other samples happen to arrive beside it.

Here the first feature has mean 4 and population standard deviation about 1.633. A future value 8 therefore becomes `(8 - 4) / 1.633`, about 2.449. It need not lie between -1 and 1: standardization is not clipping. Our constant second column uses scale 1. A future value 12 becomes 2 rather than zero, exposing a change from the training value 10. The zero-spread policy prevents division by zero; it does not prove future inputs will remain constant. See [torch.std](https://docs.pytorch.org/docs/2.8/generated/torch.std.html) and [broadcasting rules](https://docs.pytorch.org/docs/2.8/notes/broadcasting.html).

## One fitted transformation, wherever the sample arrives

Keep the original mean and scale. Do not refit on a prediction batch, even if the code still produces the expected shape.

```python
held_out = torch.tensor([[8.0, 12.0], [10.0, 14.0]])
held_normalized = (held_out - mean) / scale
alone = (held_out[:1] - mean) / scale
assert torch.equal(alone, held_normalized[:1])
assert abs(held_normalized[0, 0].item() - 2.44949) < 1e-5
assert held_normalized[:, 1].tolist() == [2.0, 4.0]

# A bug: this refits the representation on the held-out batch.
wrong = (held_out - held_out.mean(dim=0, keepdim=True))
wrong /= held_out.std(dim=0, correction=0, keepdim=True)
assert wrong.shape == held_normalized.shape
assert not torch.allclose(wrong, held_normalized)
print("Stored training transform:", held_normalized.tolist())
print("Incorrect batch-fitted transform:", wrong.tolist())
```

The incorrect version maps the first row to `[-1, -1]`. The correct version maps it to approximately `[2.449, 2]`. Shape checks alone cannot distinguish them. A useful behavioral check is **batch independence**: one sample must get the same transformed values alone or alongside other samples.

Persist the preprocessing contract as simple JSON: feature order, dtype, fitted values, and the constant-column policy. This is small metadata, so converting these two statistics to Python lists is appropriate. The examples create a named teaching artifact; rerunning this cell refreshes it. Set `AI_LAB_ARTIFACT_DIR` to choose its output directory.

```python
import json
import os
from pathlib import Path

artifact_dir = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", "artifacts/projects"))
artifact_dir.mkdir(parents=True, exist_ok=True)
normalizer_path = artifact_dir / "ch04-normalizer.json"
settings = {
    "kind": "standardize", "feature_order": ["word_count", "illustrative_score"],
    "dtype": "float32", "mean": mean.tolist(), "scale": scale.tolist(),
    "fit_partition": "train", "zero_spread_scale": 1.0,
}
normalizer_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
restored_settings = json.loads(normalizer_path.read_text(encoding="utf-8"))
restored_mean = torch.tensor(restored_settings["mean"], dtype=torch.float32)
restored_scale = torch.tensor(restored_settings["scale"], dtype=torch.float32)
restored_output = (held_out - restored_mean) / restored_scale
assert torch.equal(restored_output, held_normalized)
print("Saved and reused:", normalizer_path.resolve())
```

This contract is for the synthetic document features, not every later dataset. The digits project uses its known measurement maximum of 16 as a fixed divisor; no fitted statistics are needed for that choice. Both approaches require training and inference to use exactly the same rule. Changing a transformation means changing the model's input representation, not merely tidying a loader.

## An index can share storage

A new Python name is not proof of independent numbers. Basic tensor slicing can create a **view** that refers to existing storage. Use `clone()` when you need a separate numerical copy.

```python
source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
view = source[:, 0]
independent = view.clone()
view[0] = -1.0
assert source[0, 0].item() == -1.0
assert independent[0].item() == 1.0
print("Slicing shared storage; clone preserved its earlier values.")
```

`.item()` extracts a single Python number. `.tolist()` is useful for inspecting a small result, not for moving a large training computation back into Python loops. A reshape changes how values are arranged into dimensions; it cannot repair a mistaken interpretation of those values. Some reshapes share storage and some require a copy, so do not use reshaping as an ownership guarantee.

## Exercise — Validate and center a feature table

Implement `center_columns(x)` for a nonempty two-dimensional floating-point tensor. Reject the wrong number of dimensions, an empty dimension, or integer storage with `ValueError`. Return a new tensor with each column's mean subtracted; preserve dtype/device and do not change `x`. No standard-deviation scaling is required. The caller promises finite values.

```python exercise
def center_columns(x):
    raise NotImplementedError("Validate shape/dtype, then subtract means")

table = torch.tensor([[1.0, 8.0], [3.0, 8.0]])
centered = center_columns(table)
assert torch.equal(centered, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
assert torch.equal(table, torch.tensor([[1.0, 8.0], [3.0, 8.0]]))
```

## SOLO - fit once, transform twice, diagnose the bug

Implement `fit_standardizer(train_x)` and `apply_standardizer(x, state)`. The fit function accepts the same nonempty floating-point table as `center_columns`; reject invalid dimensionality, empty dimensions, or integer storage with `ValueError`. Return `{"mean": ..., "scale": ...}` tensors of shape `[1, feature_count]`, using population standard deviation and replacing zero spread by 1. Preserve dtype/device and do not edit the input.

The apply function accepts a floating-point `[samples, features]` table (zero samples is allowed) and a state produced by the fit function. Reject wrong dimensionality, mismatched width, integer storage, or a dtype/device different from the state with `ValueError`. Return `(x - mean) / scale` without refitting or mutating anything. The caller promises finite inputs and an unmodified state from your fit function; this is not a general artifact validator.

**Hints:** reuse the worked `dim=0, keepdim=True` expressions. Validate shape before indexing it. `torch.where` selects the safe denominator for constant columns. Empty prediction batches should produce an empty result of the same shape.

```python exercise
def fit_standardizer(train_x):
    raise NotImplementedError("Fit one training statistic per feature")

def apply_standardizer(x, state):
    raise NotImplementedError("Reuse the state; never refit here")

training = torch.tensor([[1.0, 8.0], [3.0, 8.0]])
future = torch.tensor([[5.0, 9.0], [7.0, 10.0]])
state = fit_standardizer(training)
result = apply_standardizer(future, state)
assert torch.equal(result, torch.tensor([[3.0, 1.0], [5.0, 2.0]]))
assert torch.equal(apply_standardizer(future[:1], state), result[:1])
assert apply_standardizer(torch.empty(0, 2), state).shape == (0, 2)
```

Before running, predict the two outputs in the assertion. Then deliberately replace the saved mean with `future.mean(dim=0, keepdim=True)` and explain why a shape-only test misses the defect. Add a test that catches it. Also compare `training.mean(dim=1, keepdim=True)` with `dim=0`: subtracting row means has the right output shape but mixes unlike features instead of centering each feature across samples. Explain which operation your model input contract requires.

**Optional study:** predict the shapes of `x.mean(dim=1)`, `x[:, :1]`, and `x.T` before running them. Which operation could accidentally normalize each sample instead of each feature?

**Completion gate:** explain `[samples, features]`, fit and reuse the normalizer, demonstrate batch independence and saved-settings equivalence, diagnose the held-out refitting error, and reproduce a view/clone difference. Next, autograd will track calculations on these tensors and tell us how a parameter should change.

References: [PyTorch tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html), [tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html). Answer: `solutions/draft/ch04.py`; tests: `tests/draft/test_ch04.py`.

---

## Study 05 — Autograd: make one parameter learn

New first-draft study. Source notebook: `book/draft/05-autograd.ipynb`.

# Study 05 — Autograd: make one parameter learn

**Outcome:** calculate and check a gradient, predict the effect of an update, and diagnose a training loop that moves the wrong way. PyTorch is required; no dataset or GPU is needed. We keep one scalar visible before the multi-parameter classifier in Chapter 6. Allow 30-45 minutes including the two essential exercises.

Our data pipeline now supplies numerical features. A classifier must learn how strongly those features should affect its scores. We will isolate one weighted feature, `w * x`, so a failed update can be explained with arithmetic rather than hidden inside a network. This synthetic squared-error example teaches the learning mechanism; it is not digit classification or evidence of task quality. A model's **parameters** are numbers that training changes. A **loss** is a scalar measurement of prediction error. To improve a parameter, we need to know how changing it would change that loss. A derivative answers this for one number; a collection of such derivatives is a **gradient**. PyTorch's automatic differentiation, or **autograd**, follows the tensor operations used in a calculation and computes those derivatives.

## Give the parameter a small job

Suppose a one-parameter model predicts `w * x`. For this experiment `x` is three, the desired answer is five, and `w` starts at two. The prediction is six, so its squared error is one.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
prediction = w * 3.0
loss = (prediction - 5.0) ** 2
loss.backward()

assert prediction.item() == 6.0
assert loss.item() == 1.0
assert w.grad.item() == 6.0
print("Prediction, loss, derivative:", prediction.item(), loss.item(), w.grad.item())
```

`requires_grad=True` asks PyTorch to track operations involving this parameter. `backward()` traverses the recorded calculation and accumulates the derivative into `w.grad`. The derivative is positive here: increasing `w` a little would increase the loss. A small step in the opposite direction should help.

For the function `L(w) = (3w - 5)^2`, ordinary calculus gives `L'(w) = 2(3w - 5) * 3`, which is six at `w=2`. You do not need to derive every layer by hand; this small calculation provides a check on what the tool is doing.

## Check a derivative without another differentiation system

A central **finite difference** estimates a slope by evaluating the loss on either side of a point. It is approximate and depends on the step size, but it is a useful independent debugging check.

```python
def scalar_loss(value):
    return (3.0 * value - 5.0) ** 2

epsilon = 1e-5
estimate = (scalar_loss(2.0 + epsilon) - scalar_loss(2.0 - epsilon)) / (2 * epsilon)
assert abs(estimate - w.grad.item()) < 1e-6
print("Finite-difference estimate:", round(estimate, 6))
```

At this scale Python's double-precision float is convenient. Extremely small finite-difference steps can lose accuracy through subtraction; extremely large steps measure behavior too far from the point. Finite differences are an occasional check, not the way we will train large models.

## Update and rebuild the calculation

The **learning rate** controls step size. We move against the derivative, then calculate a fresh prediction and loss. Parameter updates should not themselves become part of the differentiated training calculation, so we put the manual update inside `torch.no_grad()`.

```python
with torch.no_grad():
    w -= 0.1 * w.grad
w.grad = None

next_loss = (w * 3.0 - 5.0) ** 2
next_loss.backward()
assert next_loss.item() < 1.0
print("Updated parameter:", round(w.item(), 3))
print("New loss:", round(next_loss.item(), 3))
```

Gradients accumulate by default. If we kept the old derivative, the next update could mix gradients from separate steps. Assigning `None` clears this scalar's gradient; optimizers provide `zero_grad()` for their parameters. Accumulation can be intentional across several batches, but it must be a deliberate training policy.

We also ran a new forward calculation before the second backward pass. Do not repeatedly call `backward()` on the same saved loss merely to simulate another training step. Training means changing parameters and reevaluating their current predictions. A large learning rate can overshoot; a tiny one can make progress slow.

## Predict whether a step helps

A correct gradient does not guarantee a useful learning rate. For our fixed input `x=3`, define the prediction error as `e = 3w - 5`. The derivative is `6e`, so one update gives `w_new = w - learning_rate * 6e`. Multiplying by 3 and subtracting the target shows that the next error is `(1 - 18 * learning_rate) * e`.

That factor makes a concrete prediction. At rate 0.05 it is 0.1: the error becomes one tenth as large. At 0.1 it is -0.8: the error changes sign but shrinks, so oscillation can still converge. At 0.2 it is -2.6: it overshoots farther each time. This threshold belongs to this particular input and loss, not a universal recommended learning rate. Larger feature scales change the derivative and help explain Chapter 4's interest in normalization.

```python
for rate in (0.05, 0.1, 0.2):
    next_value = 2.0 - rate * 6.0
    next_error = 3.0 * next_value - 5.0
    print("rate", rate, "new w", round(next_value, 3),
          "new loss", round(next_error ** 2, 4))
```

The observed one-step losses are 0.01, 0.64 and 6.76, from an initial loss of 1. The last result is a deliberately disappointing update. Blaming automatic differentiation would miss the actual cause: a step too large for the local curvature.

## Reproduce the stale-gradient mistake

Now keep a reasonable rate, but omit gradient clearing. Predict the second update before running this intentionally broken two-step loop.

```python
buggy = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
trace = []
for step in range(2):
    current_loss = (buggy * 3.0 - 5.0) ** 2
    current_loss.backward()  # BUG: old gradient remains on step 2.
    trace.append((buggy.item(), current_loss.item(), buggy.grad.item()))
    with torch.no_grad():
        buggy -= 0.1 * buggy.grad
print("Before each update (w, loss, accumulated gradient):", trace)
print("Broken final value:", round(buggy.item(), 4))
assert abs(buggy.item() - 1.28) < 1e-10
```

At `w=1.4`, the fresh derivative should be -4.8. Instead, `.grad` contains the old 6 plus -4.8, or +1.2; subtracting it moves the parameter away from the target again. A new forward pass alone does not clear the old gradient buffer. The fix is to reset that buffer for each independent update. The next exercise requires implementing and testing that repair across multiple steps.

## Exercise — Implement one visible learning step

Write `gradient_step(value, target, learning_rate)`. Create a scalar floating-point tensor from `value` with gradient tracking. For this exercise the prediction is the parameter itself and the loss is `(parameter - target) ** 2`. Differentiate, take one no-grad step, and return the updated value as a Python float. Require a positive learning rate; reject zero/negative rates with `ValueError`. The caller supplies finite numerical arguments.

```python exercise
def gradient_step(value, target, learning_rate):
    raise NotImplementedError("Differentiate and update one scalar")

updated = gradient_step(3.0, 1.0, 0.1)
assert abs(updated - 2.6) < 1e-6
assert (updated - 1.0) ** 2 < (3.0 - 1.0) ** 2
```

## SOLO - repair the loop and choose a rate

Implement `fit_gain(value, input_value, target, learning_rate, steps)`. Keep one float64 scalar parameter across the run. At each step, rebuild the squared loss `(parameter * input_value - target) ** 2`, clear stale gradients, differentiate and update without tracking the update operation. Return `(final_value, losses)`, where `losses` contains the loss **before any update and after each update** (length `steps + 1`), all Python floats. Reject a nonpositive learning rate or nonpositive number of steps with `ValueError`; the caller supplies finite scalar arguments and an integer step count. No tensor input is mutated.

**Hints:** clearing before each `backward()` is sufficient. Use a new forward calculation after updates when recording the final loss. Use `.item()` rather than storing graph-connected tensors in the history. You may use the worked scalar arithmetic to predict expected results, but your implementation must use autograd.

```python exercise
def fit_gain(value, input_value, target, learning_rate, steps):
    raise NotImplementedError("Repair the repeated autograd update")

final_value, losses = fit_gain(2.0, 3.0, 5.0, 0.1, 2)
assert abs(final_value - 1.88) < 1e-10
assert len(losses) == 3
assert abs(losses[-1] - 0.4096) < 1e-10
for rate in (0.05, 0.1, 0.2):
    final_value, losses = fit_gain(2.0, 3.0, 5.0, rate, 8)
    print(rate, losses[0], losses[-1])
```

Before executing, classify each rate as shrinking steadily, shrinking with oscillation, or diverging. Use the observed losses to select one rate for this fixture and explain your decision. Then remove gradient clearing in a temporary copy and show which two-step assertion catches the bug. A decreasing loss is useful evidence, but matching the predicted two-step result is a stronger check of this exact update policy.

**Optional study:** repeat with `input_value=6`. Recompute the error multiplier before choosing a rate; do not assume the same setting remains suitable.

**Completion gate:** compare a derivative with the finite-difference estimate, implement both exercises, predict and verify the three learning-rate outcomes, and show that your repaired loop rejects the stale-gradient trajectory. Next we let `nn.Module` hold multiple parameters and an optimizer update them together.

References: [automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Answer: `solutions/draft/ch05.py`; tests: `tests/draft/test_ch05.py`.

---

## Study 06 — Train your first PyTorch classifier

New first-draft study. Source notebook: `book/draft/06-first-classifier.ipynb`.

# Study 06 - Train a classifier that reads handwritten digits

Our correspondence application receives scanned forms as well as text. One useful component would read an isolated handwritten digit in a reference-number box. The input to this component is an already cropped, normalized image; its output is a digit from 0 to 9. Finding boxes on a page and recognizing a whole number are separate jobs. This chapter trains the small recognition component so we can inspect what learning actually changes.

You already know how to align features and targets, collect batches, and calculate gradients. We will first make four easy points learn, then train a linear baseline and a small neural network on real handwritten digits. The result is a reproducible comparison with inspected mistakes, not a claim that we can process arbitrary scans.

**Setup:** use a Python environment with PyTorch, NumPy, scikit-learn and matplotlib. From the repository root, `python -m pip install -e ".[book,dev]" matplotlib` supplies the book dependencies and plotting support if needed. Select that environment as the notebook kernel. The data ships with scikit-learn; no dataset download, GPU or paid service is required. The bounded CPU experiment takes seconds to a minute on typical development machines. Run worked cells in order; cells tagged `exercise` are intentionally unfinished. Allow 60-90 minutes with practice.

## From a category to a learning signal

A **classifier** chooses a category. Digit IDs are names encoded as integers: predicting 8 when the answer is 9 is not automatically a smaller mistake than predicting 1. We therefore produce one score per class instead of fitting a single numerical answer.

Those scores are **logits**, unrestricted numbers whose relative sizes matter. Softmax converts them to nonnegative probabilities summing to one. For two logits `[0, 0]`, the probabilities are `[0.5, 0.5]`; increasing the correct class's logit raises its assigned probability. Classification chooses the largest score, which is also the largest softmax probability.

Accuracy counts correct choices, but a small score change often leaves the winning class unchanged. We train with a smoother **loss**: cross-entropy is the negative natural logarithm of the probability assigned to the correct class. Correct-class probabilities of 0.5 and 0.9 give losses about 0.693 and 0.105. Confident wrong answers cost more. PyTorch's [cross-entropy implementation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) combines the necessary calculations stably: pass raw logits, not probabilities. Here its input is `[batch, classes]` and its target is a `[batch]` tensor of integer class IDs.

An **optimizer** turns gradients into parameter updates. Stochastic gradient descent (SGD) subtracts the learning rate times each parameter's gradient. The learning rate sets the step size; too small may learn slowly, while too large may overshoot. An **epoch** is one traversal of the training data. Our first example has one batch per epoch, so each epoch makes one update.

```python
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(7)
tiny_x = torch.tensor([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]])
tiny_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
tiny_model = nn.Linear(2, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.2)
history = []

tiny_model.train()
for epoch in range(40):
    optimizer.zero_grad(set_to_none=True)
    logits = tiny_model(tiny_x)
    loss = loss_fn(logits, tiny_y)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

assert logits.shape == (4, 2)
assert history[-1] < history[0]
print("First/last fixture loss:", round(history[0], 4), round(history[-1], 4))
```

`nn.Linear(2, 2)` learns a weighted sum plus a bias for each output. `model.parameters()` exposes the tensors the optimizer must update. Each iteration clears old gradients, calculates fresh scores and loss, computes derivatives with `backward()`, and changes parameters with `step()`. These last two operations are different: gradients can exist while weights remain unchanged. Chapter 5's accumulation behavior explains why we clear gradients before each ordinary update. See the [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

`.item()` saves a Python number, rather than retaining a loss tensor and its computation history. `train()` selects training behavior; `eval()` selects evaluation behavior for layers such as dropout. Neither turns gradient tracking on or off. Evaluation below separately uses `no_grad()` because it needs predictions, not derivatives.

```python
tiny_model.eval()
with torch.no_grad():
    tiny_predictions = tiny_model(tiny_x).argmax(dim=1)
assert torch.equal(tiny_predictions, tiny_y)
print("Training-fixture predictions:", tiny_predictions.tolist())
```

`argmax(dim=1)` chooses a class within each row. Four correct training answers check the mechanism, not generalization. The deliberately easy fixture has now done its job; we need examples whose answers are not embedded in one obvious rule.

## What a hidden layer adds

A linear model gives each input pixel a fixed contribution to each digit score. The boundary between any pair of classes is where their two weighted sums are equal: a straight line with two inputs, a flat hyperplane with more. That can be a strong baseline, but it cannot express every relationship.

Consider two binary measurements. Suppose class 1 means **exactly one** is on: `(0, 1)` and `(1, 0)`; class 0 means `(0, 0)` or `(1, 1)`. This is XOR. The midpoint of each class's two points is `(0.5, 0.5)`, so one straight boundary cannot put the two classes on opposite sides. More training cannot fix this representational limit.

A hidden layer first computes intermediate features. An **activation** changes them nonlinearly; ReLU keeps positive values and replaces negative values with zero. With `s = x1 + x2`, the feature `ReLU(s) - 2 * ReLU(s - 1)` is 0, 1, 1, 0 on those four inputs. A final threshold can separate them. This is an explanation with chosen weights, not a trained XOR benchmark. In training, gradients learn such intermediate features rather than our hand-writing them.

Without an activation, two affine layers collapse into one: multiplying weighted sums still produces a weighted sum plus a bias. A **multilayer perceptron (MLP)** inserts an activation between learned layers. Its hidden units may learn useful combinations of pixels, but we should not assume each unit has a clean meaning such as "loop" or "stroke." Extra capacity can help, or fit peculiarities of the training examples. Held-out results must decide. PyTorch's [model walkthrough](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) introduces these components.

## Real digits, with a protected evaluation boundary

We use the 1,797 images bundled in [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Each is an 8x8 grid with values from 0 to 16. Flattening preserves a fixed row-major order of 64 measurements. This is a small real handwriting dataset, although scikit-learn groups it among its toy datasets because of its size.

**Attribution:** E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, [UCI repository, DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), licensed CC BY 4.0. The bundled subset is the original UCI test portion. We make new educational splits within that subset; we do not reproduce the original writer-separated benchmark.

Training rows drive gradient updates. **Validation** rows help compare model choices. **Test** rows are reserved for the final assessment after choices are frozen. Looking at test mistakes while choosing width or epochs would make test results part of development. Here we create test IDs but never score that partition; Chapter 7 develops evaluation and checkpoints further.

A **stratified** split approximately preserves each digit's frequency. Row-level random splitting answers "how well do we recognize other examples from this collection?" It does not establish performance on new writers or new scanners. This loader does not give us writer IDs for a writer-group split. Real form deployment needs representative crops and an appropriate grouping policy.

```python
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 7
digits = load_digits()
row_ids = np.arange(len(digits.target))
source_ids = np.array([f"digits-{i:06d}" for i in row_ids])
train_ids, held_ids = train_test_split(
    row_ids, test_size=0.4, random_state=SEED, stratify=digits.target
)
val_ids, test_ids = train_test_split(
    held_ids, test_size=0.5, random_state=SEED,
    stratify=digits.target[held_ids],
)
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
assert len(train_ids) + len(val_ids) + len(test_ids) == len(row_ids)

# The scale comes from the documented measurement range, not a fitted statistic.
x_train = torch.tensor(digits.data[train_ids] / 16.0, dtype=torch.float32)
y_train = torch.tensor(digits.target[train_ids], dtype=torch.long)
x_val = torch.tensor(digits.data[val_ids] / 16.0, dtype=torch.float32)
y_val = torch.tensor(digits.target[val_ids], dtype=torch.long)
print("Train/validation/test:", len(train_ids), len(val_ids), len(test_ids))
print("Training class counts:", torch.bincount(y_train).tolist())
```

Dividing by 16 brings all features into `[0, 1]` without estimating anything from held-out examples. This differs from Chapter 4's fitted mean and standard deviation: if we chose that transformation, we would fit it on training rows only and reuse the stored values. Never normalize every inference batch independently. [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) uses `random_state` to reproduce the row assignment; seeding does not promise bitwise agreement across all library versions and hardware.

Before training, look at a training example. A form-processing input must preserve the same crop, pixel scale and feature order; an arbitrary photograph resized to 8x8 does not automatically meet that contract.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.imshow(digits.images[train_ids[0]], cmap="gray_r", vmin=0, vmax=16)
ax.set_title(f"Training row {train_ids[0]}: digit {y_train[0].item()}")
ax.axis("off")
plt.show()
plt.close(fig)
```

## Batches make the small loop reusable

A **minibatch** is a subset used for one update. Its gradient estimates an update direction from fewer examples, so it varies across batches. Shuffling changes which examples share an update and avoids repeatedly following one label order. An epoch visits every training example once here, including the incomplete final batch.

A `TensorDataset` supplies the aligned `(features, target)` pair at each index, the same access contract as Chapter 3's indexable dataset. A `DataLoader` assembles those samples into batches. Unlike a one-shot iterator, the loader can produce a fresh traversal each epoch, connecting Chapter 1's iterator lesson to an actual trainer. We use `num_workers=0` for a simple local process. See [PyTorch data loading](https://docs.pytorch.org/docs/2.8/data.html).

```python
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(x_train, y_train)

def make_train_loader(batch_size=64):
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0, drop_last=False,
    )

preview_loader = make_train_loader()
batch_x, batch_y = next(iter(preview_loader))
assert batch_x.shape == (64, 64)
assert batch_y.shape == (64,)
print("Batches per epoch:", len(preview_loader))
```

The first 64 in the shape counts samples; the second counts pixels. Using a separate seeded generator for the loader lets us reset its shuffle sequence for each candidate without depending on model initialization consuming random numbers.

Our MLP maps 64 pixels to 32 hidden activations, then to ten logits. `nn.Module` tracks trainable submodules; `super().__init__()` initializes that machinery. `nn.Sequential` calls its layers in order. `forward` states the calculation, and `model(features)` invokes it through the module interface. Default randomly initialized weights give hidden units different starting points; starting every unit identically can leave them learning the same feature.

```python
class DigitMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10)
        )

    def forward(self, features):
        return self.layers(features)

assert DigitMLP()(batch_x).shape == (64, 10)

def evaluate(model, features, targets):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_fn(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}
```

We choose 30 epochs and learning rate 0.2 before comparing candidates. Both see the same split, scale, minibatch sequence, and number of updates. This is a controlled small experiment, not a fully tuned competition: another optimizer or budget may change their ranking. The linear baseline has fewer parameters and is worth keeping if extra complexity brings little benefit.

```python
def fit(model, epochs=30):
    loader = make_train_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    records = []
    for epoch in range(epochs):
        model.train()
        loss_total, seen = 0.0, 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(targets)
            seen += len(targets)
        validation = evaluate(model, x_val, y_val)
        records.append({
            "epoch": epoch + 1, "train_loss": loss_total / seen,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        })
    return records

models, histories, results = {}, {}, {}
for name in ("linear", "mlp"):
    torch.manual_seed(SEED)
    model = nn.Linear(64, 10) if name == "linear" else DigitMLP(32)
    records = fit(model)
    models[name], histories[name] = model, records
    results[name] = evaluate(model, x_val, y_val)
    parameters = sum(p.numel() for p in model.parameters())
    print(name, "parameters", parameters,
          "first/last train loss", round(records[0]["train_loss"], 4),
          round(records[-1]["train_loss"], 4),
          "validation", {k: round(v, 4) for k, v in results[name].items()})
```

Multiplying each mean batch loss by its sample count gives the final short batch its proper weight. An unweighted average of batch means would treat a 54-example batch as if it contained 64. Training loss is measured during updates; validation loss uses one fixed model after the epoch, so these are not exactly the same measurement procedure.

## Read the result before changing the model

Accuracy is a useful first metric because these ten classes have similar counts and our initial task treats a wrong digit equally regardless of its identity. It cannot show which digit pairs cause trouble. Loss additionally distinguishes tentative correct answers from confidently wrong ones.

**Recorded CPU run:** Python 3.12.9, PyTorch 2.8.0+cpu and scikit-learn 1.6.1, seed 7, 1,078 training rows and 359 validation rows:

| Model | Parameters | First / last training loss | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| Linear | 650 | 2.0301 / 0.2768 | 0.2675 | 96.66% (347/359) |
| MLP, 32 hidden units | 2,410 | 2.2416 / 0.1108 | 0.1332 | 97.49% (350/359) |

The MLP gets three more validation digits correct with about 3.7 times as many parameters. That is useful evidence for this run, not proof that nonlinear models always win or that three examples establish a reliable advantage. These are observations for this configuration, not guaranteed scores or a requirement that the MLP win. A change in runtime versions or seed may change them.

```python
fig, ax = plt.subplots(figsize=(6, 3))
for name, records in histories.items():
    ax.plot([r["epoch"] for r in records],
            [r["train_loss"] for r in records], label=f"{name}: train")
    ax.plot([r["epoch"] for r in records],
            [r["val_loss"] for r in records], "--", label=f"{name}: validation")
ax.set(xlabel="Epoch", ylabel="Cross-entropy")
ax.legend()
fig.tight_layout()
plt.show()
plt.close(fig)
```

Falling training loss means the model is fitting seen data. If validation loss rises while training loss falls, that is evidence of **overfitting**: further fitting is no longer helping unseen examples under this split. It is not a signal to inspect the test set. Here we keep the preselected epoch budget; later we can develop an explicit validation-based stopping policy.

Choose the higher validation accuracy, with the simpler linear model winning a tie. Then inspect the first three validation errors in the fixed partition order, rather than choosing the most attractive examples.

```python
selected_name = "mlp" if results["mlp"]["accuracy"] > results["linear"]["accuracy"] else "linear"
selected_model = models[selected_name]
selected_model.eval()
with torch.no_grad():
    val_predictions = selected_model(x_val).argmax(dim=1)
wrong = (val_predictions != y_val).nonzero(as_tuple=True)[0]
print("Selected:", selected_name, "validation mistakes:", len(wrong))
if len(wrong):
    shown = wrong[:3].tolist()
    fig, axes = plt.subplots(1, len(shown), figsize=(3 * len(shown), 3), squeeze=False)
    for ax, index in zip(axes[0], shown):
        row_id = int(val_ids[index])
        truth, prediction = int(y_val[index]), int(val_predictions[index])
        print(source_ids[row_id], "true", truth, "predicted", prediction)
        ax.imshow(digits.images[row_id], cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Row {row_id}: {truth} -> {prediction}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
```

The recorded MLP made nine validation mistakes. The first three were source rows `digits-001038` (9 predicted as 3), `digits-001765` (3 as 5), and `digits-000678` (5 as 4). In the first crop, the upper loop of the 9 is poorly separated at this resolution; its dark right edge and lower curve can resemble a 3. This visual observation suggests a representation problem worth investigating, but does not prove why the model erred. The other two show that errors are not confined to that digit pair. The displayed learning curves fall for both models across this budget, with no sustained rise in validation loss; that still does not establish performance on new scans. An incorrect digit can alter a reference number while still producing a plausible-looking string. Before using this component in forms, collect errors from representative crops and check whether image preparation preserves distinguishing strokes. This dataset experiment establishes a training workflow; it does not validate that new input distribution.

## SOLO - repair, train, and make a decision

### 1. Turn a broken update into a reusable training step

The function below calculates gradients but omits two essential operations. Implement `training_step(model, optimizer, features, targets)` so it selects training mode, clears old optimizer gradients, computes cross-entropy on logits, backpropagates, updates weights, and returns the **pre-update** loss as a Python float. The caller supplies aligned float features and long class IDs; rely on PyTorch for malformed tensor errors. Mutating model parameters and optimizer state is intentional; do not modify input tensors.

Before fixing it, predict what repeated calls do to the parameters and gradients. Then compare a cloned parameter before and after your repaired call. A nonempty `.grad` alone is not proof of learning. **Hint:** the missing operations belong before the forward pass and after `backward()`.

```python exercise
def training_step(model, optimizer, features, targets):
    model.train()
    loss = nn.functional.cross_entropy(model(features), targets)
    loss.backward()
    # SOLO: repair the missing operations.
    raise NotImplementedError("Finish the update and return its pre-update loss")

exercise_model = nn.Linear(2, 2)
exercise_optimizer = torch.optim.SGD(exercise_model.parameters(), lr=0.2)
before = exercise_model.weight.detach().clone()
first_loss = training_step(exercise_model, exercise_optimizer, tiny_x, tiny_y)
assert isinstance(first_loss, float)
assert not torch.equal(before, exercise_model.weight)
```

### 2. Train a smaller candidate without consulting test data

Implement `train_candidate(train_x, train_y, val_x, val_y, *, hidden=16, epochs=30, batch_size=64, seed=7)`. Return `(model, records)`, where `model` has architecture `64 -> hidden -> ReLU -> 10`, and `records` contains one dictionary per epoch with keys `epoch`, `train_loss`, `val_loss`, and `val_accuracy` as in `fit`.

Seed initialization and loader shuffling separately, use SGD with learning rate 0.2, and retain the final partial batch. Use your repaired `training_step` for each update and the worked `evaluate` for validation. All tensors are nonempty CPU tensors with the preceding shape/dtype contract; the three size arguments are positive integers. Input validation beyond that caller contract is not required. The function creates and trains a fresh model, leaves source tensors unchanged, and never needs test inputs.

**Hints:** build a `TensorDataset` from the supplied training tensors, not global variables; construct the loader once and iterate it afresh each epoch; weight mean batch losses by batch size. Copy the structure of `fit`, then change the width independently.

```python exercise
def train_candidate(train_x, train_y, val_x, val_y, *,
                    hidden=16, epochs=30, batch_size=64, seed=7):
    raise NotImplementedError("Train and evaluate a new MLP using the supplied tensors")

candidate, candidate_records = train_candidate(x_train, y_train, x_val, y_val)
assert len(candidate_records) == 30
assert candidate(x_val[:5]).shape == (5, 10)
assert candidate_records[-1]["train_loss"] < candidate_records[0]["train_loss"]
print(candidate_records[-1])
```

Record the smaller model's parameter count, final validation accuracy, loss, and at least one error ID. Compare it with both worked candidates. Does a width change help enough to justify its cost? A worse score is a valid outcome; explain it without changing the split or silently increasing the budget. The loss-decrease check is a basic learning check for this exercise, not evidence that one architecture must beat another. Keep the test partition untouched while doing this practice.

Reference answers and reasoning are in `solutions/draft/ch06.py`; tests are in `tests/draft/test_ch06.py`. Compare only after your own implementation and explanation.

## What we carry forward

The reusable result is a trained model plus its input contract and evaluation decision. Keep those together: weights without pixel order, scale, digit IDs, and architecture do not define a usable recognizer. The following in-memory metadata records the worked experiment; it does not yet save or reload anything. Source IDs such as `digits-001038` preserve the zero-based row identity in this bundled dataset; they are metadata, never model features.

```python
run_contract = {
    "task_id": "digits.v1", "dataset": "sklearn.datasets.load_digits",
    "dataset_rows": len(row_ids), "source_doi": "10.24432/C50P49",
    "source_license": "CC-BY-4.0",
    "seed": SEED, "split_policy": "stratified rows, approximately 60/20/20",
    "train_ids": source_ids[train_ids].tolist(),
    "val_ids": source_ids[val_ids].tolist(),
    "test_ids": source_ids[test_ids].tolist(), "pixel_shape": [8, 8],
    "feature_order": "row-major", "pixel_divisor": 16.0,
    "label_to_id": {str(i): i for i in range(10)}, "model": selected_name,
    "hidden": 32 if selected_name == "mlp" else None,
    "epochs": 30, "batch_size": 64, "optimizer": "SGD", "learning_rate": 0.2,
    "validation": results[selected_name], "test_evaluated": False,
    "torch_version": str(torch.__version__), "sklearn_version": sklearn.__version__,
}
print("Handoff:", run_contract["model"], "test evaluated:", run_contract["test_evaluated"])
```

Chapter 7 develops held-out evaluation and persisted checkpoints, including the difference between restoring inference and resuming training. Digit labels here are separate from the correspondence categories in the earlier Python exercises; this optional form component must not redefine their label mapping.

**Completion gate:** explain why logits and integer class targets fit this task; explain what ReLU adds that a second linear layer cannot; demonstrate a real parameter update and stale-gradient repair; train the 16-unit candidate with every training row; and justify a model choice using validation evidence and an inspected mistake. Report the reserved test boundary and the limits of this row-level split. Optional GPU work or larger architectures do not block this gate.

---

## Study 07 — Measure, split, and reproduce

New first-draft study. Source notebook: `book/draft/07-evaluation-and-checkpoints.ipynb`.

# Study 07 - Keep the result: evaluation, checkpoints, and exact resume

The digit recognizer now learns, but a notebook session is not a durable component. A form-processing job needs the same pixel transformation and label meaning tomorrow. A training job interrupted overnight should continue from its last completed epoch instead of quietly restarting its shuffle sequence or forgetting how its optimizer was moving.

This chapter turns Chapter 6's experiment into two artifacts: an **inference bundle** that reconstructs predictions in a new Python process, and a **training checkpoint** that reproduces the next updates. We also make a validation decision before opening the test partition. These are different obligations: restoring predictions does not prove training resumed correctly, and reproducing training does not prove the model generalizes.

**Setup:** use the repository's book environment with PyTorch, NumPy and scikit-learn. All data comes from bundled `load_digits`; no download, GPU or paid API is needed. Run worked cells in order. The complete CPU experiment runs three 30-epoch trajectories (linear, uninterrupted MLP, and 10+20 resumed MLP), typically within a minute. It writes a new run directory under `AI_LAB_ARTIFACT_DIR`, default `artifacts/projects`, and prints its path. Keep that directory to reuse the result. Allow 60-90 minutes with the essential resume exercise.

## Preserve the experiment before improving it

A **split** is part of the experiment, not a convenience to redraw until a score looks better. Training labels drive updates; validation labels guide choices; test labels measure the frozen decision. If the same writer or near-duplicate document appears on both sides, disjoint row IDs alone cannot guarantee independent evaluation. Our loader supplies no writer IDs, so this chapter retains the educational row split and its limitation.

We use Chapter 6's exact seed 7, 1,078/359/360 train/validation/test rows, row-major 64-pixel features, and fixed divisor 16. This measurement-scale constant learns nothing from held-out data. Chapter 4's fitted mean/scale is a different preprocessing choice; silently substituting it would change the recognizer's input contract.

**Data attribution:** E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, [UCI DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. [scikit-learn's bundled subset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) contains 1,797 images from the original UCI test portion. Our new splits do not reproduce its original writer-separated benchmark.

```python
import hashlib
import json
import os
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
SEED = 7
digits = load_digits()
indices = np.arange(len(digits.target))
train_ids, held_ids = train_test_split(
    indices, test_size=0.4, random_state=SEED, stratify=digits.target)
val_ids, test_ids = train_test_split(
    held_ids, test_size=0.5, random_state=SEED,
    stratify=digits.target[held_ids])
assert (len(train_ids), len(val_ids), len(test_ids)) == (1078, 359, 360)
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
source_ids = np.array([f"digits-{i:06d}" for i in indices])

def tensors(rows):
    return (torch.tensor(digits.data[rows] / 16.0, dtype=torch.float32),
            torch.tensor(digits.target[rows], dtype=torch.long))

x_train, y_train = tensors(train_ids)
x_val, y_val = tensors(val_ids)
# Do not construct or evaluate test tensors during development.
data_contract = {
    "task_id": "digits.v1", "source": "sklearn.datasets.load_digits",
    "source_doi": "10.24432/C50P49", "license": "CC-BY-4.0",
    "source_rows": 1797, "seed": SEED,
    "train_ids": source_ids[train_ids].tolist(),
    "validation_ids": source_ids[val_ids].tolist(),
    "test_ids": source_ids[test_ids].tolist(),
    "scaler": {"kind": "fixed_divisor", "divisor": 16.0, "input_range": [0, 16]},
    "pixel_shape": [8, 8], "feature_order": "row-major", "dtype": "float32",
    "label_to_id": {str(i): i for i in range(10)},
}
source_bytes = digits.data.astype("<f4").tobytes() + digits.target.astype("<i8").tobytes()
data_contract["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
run_id = "ch07-" + uuid.uuid4().hex[:12]
run_dir = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", "artifacts/projects")) / run_id
run_dir.mkdir(parents=True, exist_ok=False)
(run_dir / "data-contract.json").write_text(json.dumps(data_contract, indent=2), encoding="utf-8")
print("Run directory:", run_dir.resolve())
```

The source hash identifies the ordered pixels and targets used here; it is not a quality score. The saved IDs are strings for traceability, never model inputs. A new run directory preserves earlier artifacts instead of overwriting a published model.

## Why optimizer state belongs in a checkpoint

Plain SGD subtracts a scaled gradient. **Momentum** also keeps a running direction from earlier gradients, so the next update depends on both today's gradient and an optimizer buffer. For the default SGD form used here, the buffer follows `buffer = momentum * buffer + gradient`, then the update subtracts `learning_rate * buffer`. Retaining the direction can accelerate progress, but also overshoot. It does not guarantee higher accuracy.

We deliberately add momentum 0.9 and use learning rate 0.05 for this chapter, fixed before looking at results. This is a declared new training policy, not a reproduction of Chapter 6's plain-SGD scores. Both the linear baseline and MLP use this same policy and 30 epochs. Keeping nonempty momentum buffers makes an incomplete resume observable. [PyTorch's SGD documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html) specifies the update and stored state.

```python
config = {
    "kind": "mlp", "input_features": 64, "classes": 10, "hidden": 32,
    "learning_rate": 0.05, "momentum": 0.9, "batch_size": 64, "seed": SEED,
}
TOTAL_EPOCHS = 30
PAUSE_AFTER = 10

def build_model(config):
    if config["kind"] == "linear":
        return nn.Linear(config["input_features"], config["classes"])
    if config["kind"] != "mlp":
        raise ValueError("unsupported model kind")
    return nn.Sequential(
        nn.Linear(config["input_features"], config["hidden"]), nn.ReLU(),
        nn.Linear(config["hidden"], config["classes"]),
    )

def make_run(config):
    torch.manual_seed(config["seed"])
    model = build_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"])
    generator = torch.Generator().manual_seed(config["seed"])
    return model, optimizer, generator
```

Model initialization and loader shuffling have separate random-number generators. A **seed** starts a sequence; a **generator state** records the current position in that sequence. Reapplying the seed at epoch 10 would begin the shuffle sequence again. We will save both the loader generator and PyTorch's CPU generator. This particular MLP has no stochastic layers after initialization; preserving the latter still verifies the process state needed when such operations are added.

The training loop remains the one you already know. The returned loss list contains one sample-weighted mean per completed epoch. `start_epoch` counts completed epochs; `total_epochs` is the final budget, not the number of extra epochs to run.

```python
def train_epochs(model, optimizer, generator, features, targets, config,
                 start_epoch, total_epochs):
    """Return post-start epoch mean training losses, including every tail sample."""
    loader = DataLoader(TensorDataset(features, targets),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    losses = []
    for _ in range(start_epoch, total_epochs):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_y)
            seen += len(batch_y)
        losses.append(total / seen)
    return losses
```

We only checkpoint **between epochs**, after the final minibatch update. Mid-epoch continuation would additionally need the current permutation, position and any partial gradient accumulation. Our `num_workers=0`, CPU-only loop avoids worker-process and accelerator RNG state. Reproducibility across different PyTorch versions or hardware is a separate limitation; see [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

## Interrupt, restore, and compare the next updates

A `state_dict` is a mapping of state names to tensors or optimizer settings. Taking a dictionary is not necessarily an independent snapshot: its tensors can share storage with the live model. We copy state before saving. The checkpoint includes weights, optimizer buffers/settings, completed epochs, prior losses, both RNG positions, architecture and the data contract. [PyTorch's saving guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) distinguishes these from an inference-only weight file.

```python
def save_checkpoint(path, model, optimizer, generator, epoch, losses, config,
                    data_contract):
    """Snapshot all state after a complete epoch; leave no live tensor aliases."""
    checkpoint = {
        "schema_version": 1, "epoch": epoch, "losses": list(losses),
        "config": deepcopy(config), "data_contract": deepcopy(data_contract),
        "model_state": deepcopy(model.state_dict()),
        "optimizer_state": deepcopy(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().clone(),
        "loader_rng": generator.get_state().clone(),
    }
    torch.save(checkpoint, Path(path))

uninterrupted, full_optimizer, full_generator = make_run(config)
full_losses = train_epochs(uninterrupted, full_optimizer, full_generator,
                           x_train, y_train, config, 0, TOTAL_EPOCHS)
full_torch_rng = torch.get_rng_state().clone()
full_loader_rng = full_generator.get_state().clone()

interrupted, optimizer, generator = make_run(config)
first_losses = train_epochs(interrupted, optimizer, generator,
                            x_train, y_train, config, 0, PAUSE_AFTER)
checkpoint_path = run_dir / "epoch-10.pt"
save_checkpoint(checkpoint_path, interrupted, optimizer, generator,
                PAUSE_AFTER, first_losses, config, data_contract)
assert optimizer.state_dict()["state"]  # Momentum buffers really exist.
```

Now simulate another process's starting point by constructing new objects. Construction itself consumes random numbers, so restore the saved RNG state **after** initialization. Load the optimizer state into an optimizer attached to the new model's parameters, not the old model's parameters.

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
resumed, resumed_optimizer, resumed_generator = make_run(checkpoint["config"])
resumed.load_state_dict(checkpoint["model_state"])
resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
resumed_generator.set_state(checkpoint["loader_rng"])
torch.set_rng_state(checkpoint["torch_rng"])
resumed_losses = checkpoint["losses"] + train_epochs(
    resumed, resumed_optimizer, resumed_generator, x_train, y_train,
    checkpoint["config"], checkpoint["epoch"], TOTAL_EPOCHS)

assert resumed_losses == full_losses
assert all(torch.equal(a, b) for a, b in zip(uninterrupted.parameters(), resumed.parameters()))
assert torch.equal(resumed_generator.get_state(), full_loader_rng)
assert torch.equal(torch.get_rng_state(), full_torch_rng)
for before, after in zip(full_optimizer.state.values(), resumed_optimizer.state.values()):
    assert torch.equal(before["momentum_buffer"], after["momentum_buffer"])
print("30 epochs == 10 saved + 20 resumed: losses, weights, momentum, RNG positions")
```

This equality checks a stronger claim than similar accuracy: the interrupted route arrived at the same numerical training state. It is a scoped CPU test, not a guarantee for arbitrary distributed jobs. `weights_only=True` restricts the loader's supported objects; load these locally produced, trusted artifacts rather than treating that option as authentication.

## Evaluate the choice, not just the loop

The majority training label is a very cheap baseline. A trained linear model is a stronger comparison. The former reveals whether a score comes mostly from class prevalence; the latter asks whether our hidden layer earns its extra complexity under this fixed training policy.

Accuracy counts correct labels. A **confusion matrix** adds error direction: row `i`, column `j` counts true digit `i` predicted as `j`. For a particular digit, **recall** is its diagonal count divided by its row total: how many actual examples were recovered. **Precision** divides the same count by its column total: how many predictions of that digit were correct. Accuracy is reasonable for this roughly balanced ten-class experiment, but the matrix can reveal a weak digit hidden by the average. [The matrix convention](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) is also used by scikit-learn.

```python
def evaluate(model, features, targets):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        guesses = logits.argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, targets).item()
    confusion = torch.bincount(targets * 10 + guesses, minlength=100).reshape(10, 10)
    return {"loss": loss, "accuracy": (guesses == targets).float().mean().item(),
            "confusion": confusion.tolist()}, guesses

linear_config = {**config, "kind": "linear"}
linear, linear_optimizer, linear_generator = make_run(linear_config)
linear_losses = train_epochs(linear, linear_optimizer, linear_generator,
                             x_train, y_train, linear_config, 0, TOTAL_EPOCHS)
validation = {}
validation["linear"], _ = evaluate(linear, x_val, y_val)
validation["mlp"], val_guesses = evaluate(resumed, x_val, y_val)
majority_id = int(torch.bincount(y_train).argmax())
majority_validation = (y_val == majority_id).float().mean().item()
for name, report in validation.items():
    print(name, "validation loss/accuracy", round(report["loss"], 4),
          round(report["accuracy"], 4))
print("Training-majority baseline:", majority_id, round(majority_validation, 4))
selected_name = "mlp" if validation["mlp"]["accuracy"] > validation["linear"]["accuracy"] else "linear"
selected_model = resumed if selected_name == "mlp" else linear
selected_config = config if selected_name == "mlp" else linear_config
print("Frozen choice:", selected_name, "epochs:", TOTAL_EPOCHS)
```

`targets * 10 + guesses` encodes each row/column pair as an index from 0 to 99; `bincount` counts those pairs before reshaping. The feature representation, hyperparameters, epoch budget and tie rule (prefer linear) are now frozen. We did not select an epoch on test results. If you want a different budget or width, use validation and record a new experiment before final assessment.

VALIDATION_RESULT_PENDING

## SOLO - repair a resume and prove what would go wrong

Implement `resume_training(path, train_x, train_y, total_epochs)`, returning `(model, optimizer, generator, losses)`. Read a trusted checkpoint in the format above, reconstruct its configuration, restore all training state, run only the remaining epochs, and return the complete loss list including saved epochs. Reject a final budget smaller than the saved epoch with `ValueError`; equality means no further updates. The caller supplies the same ordered/scaled training rows described by the checkpoint and an integer final budget. Do not mutate those tensors. No validation or test data is needed.

**Starting point:** the worked restoration cell and `train_epochs`. **Hints:** a seed is not the saved shuffle position; call `optimizer.load_state_dict`; set RNG state after construction; use `checkpoint["epoch"]` as the loop start. Write the function independently, then change the interruption point from 10 to 5 and repeat the equality check. A different interruption boundary should not change the final model.

```python exercise
def resume_training(path, train_x, train_y, total_epochs):
    raise NotImplementedError("Restore the next update, not only predictions")

model5, optimizer5, generator5 = make_run(config)
losses5 = train_epochs(model5, optimizer5, generator5, x_train, y_train, config, 0, 5)
path5 = run_dir / "epoch-5-exercise.pt"
save_checkpoint(path5, model5, optimizer5, generator5, 5, losses5, config, data_contract)
repaired, _, _, repaired_losses = resume_training(path5, x_train, y_train, TOTAL_EPOCHS)
assert repaired_losses == full_losses
assert all(torch.equal(a, b) for a, b in zip(uninterrupted.parameters(), repaired.parameters()))
```

Deliberately omit **one** restoration operation in a temporary variant: either optimizer state or loader-generator state. Compare its parameters after one more epoch with a correctly resumed model. Explain why the first predictions immediately after loading can agree while later predictions diverge. The reference tests demonstrate both negative cases; a checkpoint file existing on disk is not enough evidence.

Complete this practice using training/validation data before running the final-test section. Published test results in this chapter are reproduction evidence, not a fresh unseen benchmark for a reader who has already studied them. Reference answers and reasoning: `solutions/draft/ch07.py`; tests: `tests/draft/test_ch07.py`.

## Open the test partition once choices are fixed

We now score the selected model and the already fixed linear baseline, without changing either based on the result. The baseline is reported even if it loses. For a finite test set, differences of a few samples can reflect sampling variation; there is no arbitrary minimum accuracy gate here.

```python
x_test, y_test = tensors(test_ids)
test_report, test_guesses = evaluate(selected_model, x_test, y_test)
linear_test, _ = evaluate(linear, x_test, y_test)
majority_test = (y_test == majority_id).float().mean().item()
print("Selected test loss/accuracy:", round(test_report["loss"], 4),
      round(test_report["accuracy"], 4))
print("Frozen linear/majority test accuracy:", round(linear_test["accuracy"], 4),
      round(majority_test, 4))
confusion = torch.tensor(test_report["confusion"])
assert int(confusion.sum()) == len(y_test)
print("Rows actual, columns predicted; order 0..9:", confusion.tolist())
recalls = confusion.diag().float() / confusion.sum(dim=1)
weakest = int(recalls.argmin())
print("Lowest test recall digit:", weakest, "recall:", round(recalls[weakest].item(), 4))
errors = (test_guesses != y_test).nonzero(as_tuple=True)[0]
for i in errors[:3].tolist():
    print(source_ids[test_ids[i]], "actual", int(y_test[i]), "predicted", int(test_guesses[i]))
```

TEST_RESULT_PENDING

Reading these errors describes the frozen assessment; using them to tune again would spend the test set. This small row-split evaluation still does not establish performance on new writers, new form layouts or different crop preparation. A wrong digit can silently change a reference number, so application-level evaluation must include whole-field correctness as well as single-digit accuracy.

## Save an inference bundle and cross a process boundary

Inference needs weights, architecture, feature order, scaling and label mapping. It does not need momentum buffers. Save these separately from the training checkpoint so a prediction process has a small, explicit dependency boundary. The bundle also records the splits, source identity, training policy and observed metrics. Its file hashes detect accidental changes; they do not establish trust in a bundle from an unknown source.

```python
bundle_dir = run_dir / "inference"
bundle_dir.mkdir()
torch.save(selected_model.state_dict(), bundle_dir / "weights.pt")
# Two validation crops are a restore probe, not a new training or test partition.
probe = {"source_ids": source_ids[val_ids[:2]].tolist(),
         "pixels": digits.data[val_ids[:2]].tolist()}
(bundle_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
manifest = {
    "schema_version": 1, "artifact_id": f"digits-{selected_name}-{run_id}",
    "task_id": "digits.v1", "created_at": datetime.now(timezone.utc).isoformat(),
    "config": selected_config, "epochs": TOTAL_EPOCHS, "data": data_contract,
    "training_policy": "sgd-momentum-v1", "validation": validation[selected_name],
    "test": test_report, "linear_test": linear_test, "majority_test": majority_test,
    "versions": {"torch": str(torch.__version__), "sklearn": sklearn.__version__},
    "files": {name: hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest()
              for name in ("weights.pt", "probe.json")},
}
(bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

The next Python process imports no notebook or answer module and performs no training. It reconstructs the architecture from JSON, verifies the two consumed files, reads the raw probe pixels, applies the saved divisor and produces logits. The small reconstruction code is shown completely so the new process does not depend on a hidden notebook global.

```python
restore_code = r'''
import hashlib, json, sys
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(1)
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for name in ("weights.pt", "probe.json"):
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != manifest["files"][name]:
        raise ValueError("Artifact checksum mismatch: " + name)
cfg = manifest["config"]
if cfg["kind"] == "linear":
    model = nn.Linear(cfg["input_features"], cfg["classes"])
else:
    model = nn.Sequential(nn.Linear(cfg["input_features"], cfg["hidden"]),
                          nn.ReLU(), nn.Linear(cfg["hidden"], cfg["classes"]))
model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
probe = json.loads((root / "probe.json").read_text(encoding="utf-8"))
scaler = manifest["data"]["scaler"]
if scaler["kind"] != "fixed_divisor":
    raise ValueError("Unsupported preprocessing")
features = torch.tensor(probe["pixels"], dtype=torch.float32) / scaler["divisor"]
model.eval()
with torch.no_grad():
    print(json.dumps(model(features).tolist()))
'''
child = subprocess.run([sys.executable, "-I", "-c", restore_code, str(bundle_dir.resolve())],
                       check=True, capture_output=True, text=True, timeout=60)
restored_logits = torch.tensor(json.loads(child.stdout), dtype=torch.float32)
selected_model.eval()
with torch.no_grad():
    original_logits = selected_model(x_val[:2])
assert torch.equal(restored_logits, original_logits)
print("Fresh-process logits identical; durable bundle:", bundle_dir.resolve())
```

The `-I` option starts Python in isolated mode, excluding notebook-specific Python path settings. The process still uses the selected environment's installed dependencies. Exact logits show that these saved weights and preprocessing reproduce the two probes in this CPU environment; they do not test every possible input. The test suite separately verifies the resume path and deliberately broken state restoration.

**Optional metric extension:** retain the earlier `binary_counts(predictions, targets)` exercise for a two-class task: equally sized binary lists, class 1 positive, reject misalignment/nonbinary entries with `ValueError`, return integer `tp/fp/fn/tn`, and return zeros for empty lists. Explain the zero-denominator policy before deriving precision/recall. This helper alone does not complete this chapter.

**Completion gate:** preserve the approved split and input contract; explain what validation chose before test evaluation; interpret at least one off-diagonal confusion entry; implement the changed-boundary resume and its deliberate failure; and demonstrate a new process reproducing logits from the bundle alone. You now own both a restartable training run and a restorable prediction artifact. The next pretrained-model studies reuse the same train/evaluate/save discipline while adding tokenizers and pretrained weights.

---

## Study 08 — Use a pretrained tokenizer and model

New first-draft study. Source notebook: `book/draft/08-pretrained-model-inputs.ipynb`.

# Study 08 - Give a pretrained model the inputs it expects

Our correspondence router now needs to read text. The digit model from Chapters 6–7 cannot consume a message, and the early `claim`/`invoice` examples were synthetic Python fixtures. From here the real teaching task is `sms-routing.v1`: classify public SMS messages as legitimate (`ham=0`) or unwanted (`spam=1`). An operator may later send uncertain predictions to `review`; that is an operational route, not a third training label.

This chapter builds the input side of that system. We will run a real pretrained tokenizer and encoder, inspect the resulting shapes, measure how much text a length limit removes, and save the tokenizer with its provenance. Chapter 9 adds a classifier and learns from labels. No classification quality is implied by obtaining a vector.

**Starting point:** tensors, batches, evaluation mode and saved artifacts from Chapters 4–7. Use the repository's CPU book environment (`python -m pip install -e ".[book,dev]"`) as your notebook kernel. Allow 45–60 minutes with practice; the worked cached-model run is small. Run worked cells in order and leave tagged exercises unexecuted until you attempt them. No paid service or GPU is required.

## A token ID is an address; a representation is a learned vector

A tokenizer splits text into vocabulary entries and returns their integer IDs. BERT uses word pieces: a rare word can become several familiar fragments. Special entries mark sequence boundaries and padding. The tokenizer's normalization and vocabulary must match the encoder's training. Replacing it with `text.split()` would change the meaning of every input address.

An embedding table looks up a vector for each ID. ID 20 is not twice as positive, long, or meaningful as ID 10. The encoder then mixes information across positions, so a token's final **contextual representation** depends on surrounding tokens. The same word in two sentences can have the same ID but different output vectors. Pretraining learned these parameters before our SMS task; it did not learn our two-class head. The [BERT paper](https://arxiv.org/abs/1810.04805) describes masked-word pretraining and downstream adaptation. We need its input contract now; implementing attention from scratch can wait until the architecture study.

Here is a deliberately synthetic vocabulary that makes the addressing visible. It is not BERT's vocabulary.

```python
vocabulary = {"[PAD]": 0, "[UNK]": 1, "message": 2, "received": 3}
sequences = [[2, 3], [2], [1, 1]]
width = max(map(len, sequences))
fixture_ids = [row + [0] * (width - len(row)) for row in sequences]
fixture_mask = [[1] * len(row) + [0] * (width - len(row)) for row in sequences]
assert fixture_ids == [[2, 3], [2, 0], [1, 1]]
assert fixture_mask == [[1, 1], [1, 0], [1, 1]]
print("IDs:", fixture_ids, "mask:", fixture_mask)
```

**Padding** fills short rows so a batch is rectangular. The **attention mask** marks real positions with 1 and padding with 0. Unknown vocabulary entries are still real input; they get mask 1. **Truncation** removes tokens beyond a limit, possibly removing useful evidence. Padding adds storage; truncation loses information. For single BERT messages our limit includes `[CLS]` and `[SEP]`, leaving 62 content positions at a limit of 64. See the [padding/truncation guide](https://huggingface.co/docs/transformers/main/pad_truncation).

## Acquire once, then load verified local files

We use [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny), a converted compact BERT with two encoder layers and hidden width 128; its model card declares MIT. Our recorded snapshot revision is `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`. This small base is useful for CPU learning, not a promise of state-of-the-art SMS accuracy.

Downloads default to disabled. From the repository root, explicitly acquire the base with `python tools/prepare_book_assets.py --asset bert --allow-downloads`. The helper bounds downloads, records immutable revision/file hashes and verifies the cache. `AI_LAB_CACHE_DIR` chooses the cache root; `AI_LAB_MODEL_DIR` can identify a complete BERT snapshot with its `metadata.json`. `AI_LAB_ALLOW_DOWNLOADS=1` also permits the helper to acquire a missing asset. Framework calls below always read the resolved local path and disable custom remote code.

The dataset is Almeida and Hidalgo's (2011) [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI `10.24432/C5CC84`, CC BY 4.0. Set `AI_LAB_SMS_PATH` to its source ZIP or `SMSSpamCollection` file, or enable downloads for first preparation. The shared loader preserves original text and physical-line IDs, removes canonical duplicates, and excludes conflicting-label groups. It creates the frozen seed-42 800/200/200 partitions once; later chapters reload their saved IDs. Chapter 10 explains those decisions. This historical English SMS collection is a proxy for correspondence, not proof of performance on current email or new senders.

```python
import os
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "config/book_manifest.json").is_file())
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("USE_TF", "0")
from ai_lab.projects.assets import resolve_asset, asset_metadata
from ai_lab.projects.sms import (
    load_sms_records, make_sms_splits, save_sms_splits, load_sms_splits,
)

ARTIFACTS = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", ROOT / "artifacts/projects"))
SPLIT_DIR = ARTIFACTS / "sms-splits-v1"
ready = False
try:
    base_path = resolve_asset("bert")
    base_metadata = asset_metadata(base_path)
    if not SPLIT_DIR.exists():
        save_sms_splits(make_sms_splits(load_sms_records()), SPLIT_DIR)
    splits = load_sms_splits(SPLIT_DIR)
    ready = True
except FileNotFoundError as error:
    # A supplied but invalid location is a configuration error, never a fallback.
    if (os.environ.get("AI_LAB_RUN_PRETRAINED") == "1"
            or os.environ.get("AI_LAB_MODEL_DIR") or os.environ.get("AI_LAB_SMS_PATH")):
        raise
    print("SKIP required pretrained experiment; chapter incomplete:", error)
if ready:
    assert splits.metadata["sizes"] == {"train": 800, "validation": 200, "test": 200}
    print("Base revision:", base_metadata["revision"])
    print("Split identity:", splits.metadata["split_sha256"])
    print("Class counts:", splits.metadata["class_counts"])
```

`next(...)` finds the repository while allowing the notebook to start in its own directory. The shared helper is project preparation code, not a hidden answer: it supplies checked records and split membership, while the tokenizer and model work remains here. Missing required assets print an explicit incomplete result. Set `AI_LAB_RUN_PRETRAINED=1` for validation that must fail instead of skipping. That switch does not authorize downloads. A corrupt cache raises an error; it is not silently replaced with random weights.

## Read what the actual tokenizer produces

`AutoTokenizer` chooses a tokenizer from checkpoint configuration. `return_tensors="pt"` requests PyTorch tensors. A returned dictionary can be expanded with `**encoded`, equivalent to supplying `input_ids=...`, `attention_mask=...` and other named arguments. `token_type_ids` identifies sentence segments; our single-message inputs use segment 0 throughout.

```python
if ready:
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False)
    encoder.eval()
    # Invented safe sentences expose tokens without logging public messages.
    probe_texts = ["A message arrived.", "An unexpectedly long message arrived today."]
    probe = tokenizer(probe_texts, padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
    print("Probe tokens:", tokenizer.convert_ids_to_tokens(probe["input_ids"][0]))
    print("Input shapes:", {k: tuple(v.shape) for k, v in probe.items()})
    print("Mask:", probe["attention_mask"].tolist())
    assert probe["input_ids"].dtype == torch.long
    assert encoder.config.hidden_size == 128
    assert encoder.config.num_hidden_layers == 2
```

Now run original text from the first four **training** records. We print only IDs and shapes. Tokenization cannot learn from labels, but developing input policies by looking through the test corpus would still weaken our held-out boundary.

```python
if ready:
    messages = [row.text for row in splits.train[:4]]
    encoded = tokenizer(messages, padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
    assert hidden.shape == (*encoded["input_ids"].shape, 128)
    print("Record IDs:", [row.record_id for row in splits.train[:4]])
    print("Tokens -> contextual vectors:", tuple(encoded["input_ids"].shape),
          "->", tuple(hidden.shape))
```

The shape is `[batch, positions, hidden_width]`. Each position now has 128 floating-point features, rather than one integer address. It is not `[batch, 2]` class scores. `eval()` disables dropout's training behavior; `no_grad()` separately avoids constructing derivative graphs. Both matter for a repeatable inference comparison.

## Why the mask must also reach your own pooling code

Suppose we want one vector per message. **Mean pooling** averages its position vectors. If a two-token message has vectors `[2, 4]` and `[4, 6]`, its mean is `[3, 5]`. Appending padding should not divide that sum by a larger number. BERT can produce nonzero vectors at padded output positions: its input attention mask prevents reading padded keys, but does not turn every padded output vector into zero. A downstream mean must mask them itself.

For hidden tensor `H` and mask `M`, sum `H[b,t,:] * M[b,t]` over positions `t`, then divide by the sum of the mask for that row. `unsqueeze(-1)` gives `[batch, positions, 1]`, so broadcasting applies the mask to all 128 features. Here the mean deliberately includes special tokens; a different pooling policy is a different feature definition. This convenient representation is not a sentence-similarity model or BERT's built-in classification pooler.

```python
if ready:
    weights = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
    assert pooled.shape == (4, 128)
    # Same text, extra right padding: real positions and masked mean should agree.
    fixed = tokenizer(messages, padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
    with torch.no_grad():
        fixed_hidden = encoder(**fixed).last_hidden_state
    fixed_weights = fixed["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    fixed_mean = (fixed_hidden * fixed_weights).sum(1) / fixed_weights.sum(1)
    torch.testing.assert_close(pooled, fixed_mean, rtol=1e-4, atol=1e-5)
    print("Dynamic/fixed padded slots:", int((encoded["attention_mask"] == 0).sum()),
          int((fixed["attention_mask"] == 0).sum()))
    print("Masked mean parity passed; unmasked means equal:",
          torch.allclose(hidden.mean(1), fixed_hidden.mean(1)))
```

The equality check concerns padding, not truncation. If a shorter limit removes real tokens, both contextual features and their mean can change. Choosing 64 because the batch fits is an engineering budget; it needs a data measurement too.

```python
if ready:
    train_texts = [row.text for row in splits.train]
    lengths = [len(ids) for ids in tokenizer(train_texts, truncation=False)["input_ids"]]
    length_report = {str(limit): sum(n > limit for n in lengths) for limit in (32, 64)}
    print("Training messages exceeding each total-token limit:", length_report)
    print("Longest training message in tokens:", max(lengths))
```

**Recorded CPU run (2026-10-30):** the first four training messages produced `[4,58]` IDs and `[4,58,128]` contextual features. Dynamic padding used 92 padded slots; fixed width 64 used 116. Masked means agreed within the stated tolerance; unmasked means did not. Of 800 training messages, 223 exceeded 32 total tokens and 11 exceeded 64; the longest had 221. Thus 32 would affect 27.9% of training messages versus 1.4% at 64. These are observations for the frozen split and tokenizer, not a quality benchmark.

A count of messages affected is more useful than silently truncating, but it cannot prove the removed suffixes were unimportant. Chapter 9 holds the 64-token budget fixed for its comparison. The exercise asks you to measure the representation effect at 32; it does not grant permission to tune on test messages.

## Persist the input contract

`save_pretrained` saves the tokenizer's actual vocabulary and settings. The length and pooling choices belong to our application, so we record them separately with the verified base metadata and split identity. A new directory per run prevents overwriting an earlier handoff. The base model remains in the verified cache; Chapter 9 will save adapted weights as well.

```python
if ready:
    import json
    import tempfile
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    input_bundle = Path(tempfile.mkdtemp(prefix="sms-inputs-v1-", dir=ARTIFACTS))
    tokenizer.save_pretrained(input_bundle)
    contract = {
        "task_id": "sms-routing.v1", "label_map": {"ham": 0, "spam": 1},
        "max_length": 64, "padding": "longest", "truncation": True,
        "pooling_demo": "masked mean including special tokens",
        "base": base_metadata, "split_sha256": splits.metadata["split_sha256"],
        "train_length_report": length_report,
    }
    (input_bundle / "input-contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8")
    restored_tokenizer = AutoTokenizer.from_pretrained(
        input_bundle, local_files_only=True, trust_remote_code=False)
    reencoded = restored_tokenizer(messages, padding=True, truncation=True,
                                   max_length=64, return_tensors="pt")
    assert set(reencoded) == set(encoded)
    for key in encoded:
        assert torch.equal(encoded[key], reencoded[key])
    print("Tokenizer round trip passed:", input_bundle.name)
```

## SOLO - measure a different length policy

Implement `inspect_features(tokenizer, encoder, texts, *, max_length=64)`. For a nonempty list of nonblank strings and a total-token limit from 4 through 512, return a dictionary with `features` (a CPU `[N,128]` masked mean), `lengths` (untruncated token counts), and `truncated` (how many counts exceed the limit). Use dynamic padding, inference in evaluation mode and the supplied **real** tokenizer/encoder. Do not change texts or weights. It is acceptable to leave the encoder in evaluation mode. The caller supplies a CPU BERT model of this chapter's width and valid texts/limit; extra validation is not required.

Build on the worked tokenization, pooling and length cells. Python `sum(condition for ...)` counts true values; `.tolist()` converts a tensor into ordinary values if you need to inspect it. The important independent step is connecting those operations in one function, then evaluating a changed limit.

```python exercise
def inspect_features(tokenizer, encoder, texts, *, max_length=64):
    raise NotImplementedError("Tokenize, infer, mask-pool, and report truncation")

chosen_texts = [row.text for row in splits.train[:32]]
short = inspect_features(tokenizer, encoder, chosen_texts, max_length=32)
long = inspect_features(tokenizer, encoder, chosen_texts, max_length=64)
assert short["features"].shape == long["features"].shape == (32, 128)
assert short["truncated"] >= long["truncated"]
print("Affected messages:", short["truncated"], long["truncated"])
print("Mean absolute feature change:",
      (short["features"] - long["features"]).abs().mean().item())
```

Explain why an unchanged shape does not mean unchanged information. Find one training ID whose sequence is longer than 32 and compare its two vectors. If the first 32 contain none, choose the first such ID in the complete training partition. State whether you would keep 64 under our CPU budget and what validation experiment would be needed before shortening it. Do not infer classification improvement from a vector difference alone.

Separate answers and reasoning: `solutions/draft/ch08.py`; tests: `tests/draft/test_ch08.py`. The tests retain the small padding fixture and add real-model padding invariance and tokenizer reload checks.

**Completion gate:** execute the real encoder on public training records; explain IDs, contextual vectors, shapes and masks; demonstrate padding invariance and tokenizer restoration; implement the length comparison and justify its limits. A missing-asset SKIP leaves this gate incomplete. What we carry forward is the frozen data identity, a real BERT input contract and a reusable tokenizer—not yet a trained SMS classifier.

---

## Study 09 — Fine-tune a small pretrained model

New first-draft study. Source notebook: `book/draft/09-first-fine-tuning.ipynb`.

# Study 09 - Adapt BERT, then decide whether it earns its cost

The SMS router has real inputs and a verified pretrained encoder. It still has no learned connection between those inputs and `ham=0` / `spam=1`. Here we fit that connection, compare two update policies, and save a model that can predict after the notebook is gone. A useful outcome may be that a simpler baseline wins.

You need the minibatch loop and held-out reasoning from Chapters 6–7 and BERT inputs from Chapter 8. The complete CPU experiment uses the same 800 training, 200 validation and 200 test records, batch size 16, a 64-token limit and three epochs per candidate. That is 150 optimizer updates for each BERT policy. Allow about an hour with practice; runtime depends on CPU. Use the repository's `.[book,dev]` environment with PyTorch, Transformers and scikit-learn. No GPU or paid API is required. Worked cells are independent of tagged learner cells.

## Learn a head, or change the representation too?

A **classification head** maps an encoder's representation to our two logits. In `BertForSequenceClassification`, the base's pooled first-token representation passes through dropout and a learned linear classifier. Its output is `[batch, 2]`; integer targets are `[batch]`. Cross-entropy connects the correct class to a gradient, exactly as in the digit classifier. The public base supplies pretrained BERT parameters, but the two-class head starts randomly. Its initialization warning is expected; it is not evidence that the base was randomly initialized.

**Head-only adaptation** holds the encoder fixed and learns a boundary in its representation space. It has fewer trainable parameters and reduces the risk of overwriting useful features. But if SMS distinctions are poorly represented, the head cannot repair them. **Partial fine-tuning** also updates the final encoder layer: gradients can reshape the features while embeddings, the first layer and the pooler remain fixed. More adaptable parameters bring more compute and overfitting risk on just 800 messages. The [BERT model documentation](https://huggingface.co/docs/transformers/model_doc/bert) explains the base and classification outputs; the [training guide](https://huggingface.co/docs/transformers/training) describes supervised adaptation.

Freezing means setting `requires_grad=False` and excluding that tensor from the optimizer. `eval()` is a different switch: it changes dropout behavior but does not prevent gradients. We keep the whole BERT base in evaluation mode for both policies, disabling encoder dropout during this controlled experiment; trainable final-layer weights still receive derivatives. The head's dropout remains active during training. This is a deliberate small-data recipe, not a rule that all fine-tuning must disable encoder dropout.

## Keep data and model identity fixed

Use the same cache policy as Chapter 8. `python tools/prepare_book_assets.py --asset bert --allow-downloads` prepares the verified base. `AI_LAB_CACHE_DIR` chooses the cache root; `AI_LAB_MODEL_DIR` selects a complete base snapshot with provenance. `AI_LAB_SMS_PATH` points to the UCI ZIP/raw file, or `AI_LAB_ALLOW_DOWNLOADS=1` permits first acquisition. Framework loading below is local-only with remote code disabled. `AI_LAB_RUN_PRETRAINED=1` turns missing assets into a hard validation error; it does not permit downloads.

The source is Almeida and Hidalgo (2011), [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI `10.24432/C5CC84`, CC BY 4.0. The base is [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny), MIT, recorded revision `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`. The saved source metadata retains attribution, raw hash and duplicate policy. The deterministic split removes canonical duplicates before selection; it is neither sender-disjoint nor temporal. Test results therefore describe this historical collection, not a deployment guarantee.

```python
import os
import sys
from pathlib import Path
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "config/book_manifest.json").is_file())
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("USE_TF", "0")
from ai_lab.projects.assets import resolve_asset, asset_metadata
from ai_lab.projects.sms import (
    load_sms_records, make_sms_splits, save_sms_splits, load_sms_splits,
    TfidfFeatures, classification_metrics,
)
ARTIFACTS = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", ROOT / "artifacts/projects"))
SPLIT_DIR = ARTIFACTS / "sms-splits-v1"
ready = False
try:
    base_path = resolve_asset("bert")
    base_metadata = asset_metadata(base_path)
    if not SPLIT_DIR.exists():
        save_sms_splits(make_sms_splits(load_sms_records()), SPLIT_DIR)
    splits = load_sms_splits(SPLIT_DIR)
    assert splits.metadata["sizes"] == {"train": 800, "validation": 200, "test": 200}
    ready = True
except FileNotFoundError as error:
    if (os.environ.get("AI_LAB_RUN_PRETRAINED") == "1"
            or os.environ.get("AI_LAB_MODEL_DIR") or os.environ.get("AI_LAB_SMS_PATH")):
        raise
    print("SKIP required adaptation; chapter incomplete:", error)
if ready:
    print("Split SHA:", splits.metadata["split_sha256"])
    print("Class counts:", splits.metadata["class_counts"])
```

The shared loader handles acquisition and saved IDs; the modeling decisions remain visible below. No labels enter the tokenizer. Training alone fits parameters and learned preprocessing. Validation selects candidate and epoch; test is scored only after those choices are frozen. When you attempt the exercise, work above that final test cell and leave it unexecuted until your own choice is fixed. Published worked test results are already known, so further experiments on this public split are development exercises, not new blind tests.

## Establish what a cheap predictor can do

Imbalance makes accuracy deceptive. If 88 of 100 messages are ham, always saying ham gets 88% accuracy while catching no spam. We report a confusion matrix with true classes as rows and predictions as columns, both ordered `[ham, spam]`. A false positive quarantines a legitimate message; a false negative lets spam through. Spam precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`. F1 balances precision and recall; **macro-F1** averages the two class F1 scores equally so ham's larger count cannot dominate. Undefined ratios are reported as zero. This selection metric is educational; an operator's costs may justify a different threshold policy later.

Our second baseline uses training-fitted unigram TF-IDF and logistic regression. TF-IDF weights words by within-message frequency and how uncommon they are among training messages. It ignores word order, but many SMS cues are lexical. The shared implementation uses at most 2,048 terms with training document frequency at least two, log term counts, smoothed IDF and row normalization. Its saved vocabulary/IDF fix the meaning of each column. Logistic regression learns a linear two-class boundary with regularization; unlike BERT it learns no contextual text representation. See [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) and [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

```python
if ready:
    import torch
    import numpy as np
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    SEED = 42
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False)
    train_text = [r.text for r in splits.train]
    val_text = [r.text for r in splits.validation]
    train_y = [r.label for r in splits.train]
    val_y = [r.label for r in splits.validation]
    majority = Counter(train_y).most_common(1)[0][0]
    features = TfidfFeatures.fit(train_text, max_features=2048, min_df=2)
    baseline = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    baseline.fit(features.transform(train_text), train_y)
    baseline_validation = {
        "majority": classification_metrics(val_y, [majority] * len(val_y)),
        "tfidf_logistic": classification_metrics(
            val_y, baseline.predict(features.transform(val_text)).tolist()),
    }
    print("Validation baselines:", baseline_validation)
```

We do not rebalance or choose regularization after inspecting test results. Baselines and BERT see the same records. These two baselines are controls, not obstacles for a larger model to be forced to beat.

## Prepare batches and an auditable update policy

A list of records is a valid map-style dataset for this small run. The loader's `collate_fn` receives a list of selected records and turns it into tensors. This preserves text/label alignment while padding only the current batch. A separate seeded loader generator repeats the same batch order for both candidates.

```python
if ready:
    def collate(records):
        batch = tokenizer([r.text for r in records], padding=True,
                          truncation=True, max_length=64, return_tensors="pt")
        batch["labels"] = torch.tensor([r.label for r in records], dtype=torch.long)
        return batch

    def new_model(policy):
        if policy not in {"head", "last_layer"}:
            raise ValueError("policy must be head or last_layer")
        torch.manual_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=2, id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
            local_files_only=True, trust_remote_code=False)
        for parameter in model.bert.parameters():
            parameter.requires_grad_(False)
        if policy == "last_layer":
            for parameter in model.bert.encoder.layer[-1].parameters():
                parameter.requires_grad_(True)
        return model

    def score(model, records):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in DataLoader(records, batch_size=16, collate_fn=collate):
                predictions.extend(model(**batch).logits.argmax(1).tolist())
        return classification_metrics([r.label for r in records], predictions), predictions
```

The `[-1]` index chooses the final encoder layer without assuming its numeric name. The head remains trainable because it is outside `model.bert`. Supplying only a parameter-name list would not prove this policy ran: we will compare cloned tensors before and after actual updates.

## Three full epochs, with validation-based checkpoint selection

Use AdamW at 0.001 for the new head and 0.00005 for the final pretrained layer when enabled, with weight decay 0.01. Parameter groups allow different learning rates in one optimizer. The smaller pretrained rate limits each update to an already useful representation; it is a chosen starting recipe, not an optimal value established in advance.

We save the epoch with highest validation macro-F1, breaking ties in favor of the earlier epoch. The copy must clone each tensor: a shallow dictionary copy would still share storage with ongoing training. We train all three epochs to compare their curves, then restore that selected state. No optimizer or RNG state is saved here; this is an inference artifact, not Chapter 7's training-resume checkpoint.

```python
if ready:
    import time
    def fit_policy(policy, *, encoder_lr=5e-5):
        model = new_model(policy)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        head = list(model.classifier.parameters())
        groups = [{"params": head, "lr": 1e-3}]
        if policy == "last_layer":
            groups.append({"params": list(model.bert.encoder.layer[-1].parameters()),
                           "lr": encoder_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        loader = DataLoader(splits.train, batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED),
                            collate_fn=collate, num_workers=0)
        best_f1, best_state, best_epoch = -1.0, None, None
        history = []
        started = time.perf_counter()
        for epoch in range(1, 4):
            model.train()
            model.bert.eval()  # mode and gradient policy are independent
            total_loss, seen, updates = 0.0, 0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                n = len(batch["labels"])
                total_loss += loss.item() * n
                seen += n
                updates += 1
            metrics, _ = score(model, splits.validation)
            history.append({"epoch": epoch, "seen": seen, "updates": updates,
                            "train_loss": total_loss / seen, "validation": metrics})
            if metrics["macro_f1"] > best_f1:
                best_f1, best_epoch = metrics["macro_f1"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        changed = [name for name, p in model.named_parameters()
                   if not torch.equal(before[name], p)]
        frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
        assert not set(changed).intersection(frozen)
        assert any(name.startswith("classifier.") for name in changed)
        if policy == "last_layer":
            assert any(name.startswith("bert.encoder.layer.1.") for name in changed)
        return model, {"policy": policy, "best_epoch": best_epoch, "history": history,
                       "seconds": time.perf_counter() - started,
                       "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                   if p.requires_grad),
                       "changed": changed, "frozen": frozen,
                       "encoder_lr": encoder_lr if policy == "last_layer" else None}

    models, runs, validation = {}, {}, {}
    for policy in ("head", "last_layer"):
        models[policy], runs[policy] = fit_policy(policy)
        validation[policy], _ = score(models[policy], splits.validation)
        print(policy, runs[policy]["trainable_parameters"], "trainable parameters; epoch",
              runs[policy]["best_epoch"], "seconds", round(runs[policy]["seconds"], 2))
        print("Epoch loss/F1:", [(r["epoch"], round(r["train_loss"], 4),
               round(r["validation"]["macro_f1"], 4)) for r in runs[policy]["history"]])
        print("Changed/frozen tensors:", len(runs[policy]["changed"]),
              len(runs[policy]["frozen"]), "validation:", validation[policy])
```

Frozen assertions cover every frozen parameter, including the large embedding table and first layer. Requiring at least one final-layer tensor to change catches a disconnected optimizer group. It would be too strong to require every trainable scalar to change: some gradients may be zero. A falling training loss shows fitting, not necessarily useful generalization. Compare validation macro-F1 and error types before choosing extra capacity.

**Recorded run (2026-10-30):** Python 3.12.9, PyTorch 2.8.0+cpu, Transformers 4.54.1, scikit-learn 1.6.1, two CPU threads. Each policy visited all 800 rows three times (150 updates). Head-only changed 2 tensors and left 39 frozen; final-layer adaptation changed 18 and left 23 frozen. The trainable counts were 258 and 198,530. Measured training/validation/weight-audit time was 1.38s and 1.99s on this AMD64 workstation, excluding model loading; these are observations, not portable timing promises.

Head-only training loss fell from 0.3309 to 0.2000; its selected epoch was 3. Partial fine-tuning fell from 0.2762 to 0.0696, but validation macro-F1 peaked at epoch 2 (0.9116) then fell to 0.9051. Continuing to lower training loss therefore did not select a better validation checkpoint. Both curves and complete confusion matrices are preserved in the saved report.
## Inspect a failure before judging the winner

Choose the higher validation macro-F1, with the smaller head-only policy winning a tie. For each error type, inspect the first ID in saved validation order. The printed diagnostics contain no message text: length/truncation, whether the text contains a currency symbol, and digit count. These are clues for a hypothesis, not a causal explanation. A missed spam message without a currency cue suggests that a simple surface rule cannot recover all spam; it does not prove that BERT relies on that cue. Source text remains in the local prepared dataset for controlled investigation.

```python
if ready:
    selected = ("last_layer" if validation["last_layer"]["macro_f1"] >
                validation["head"]["macro_f1"] else "head")
    chosen = models[selected]
    _, predictions = score(chosen, splits.validation)
    inspected_errors = {}
    for name, truth, prediction in (("false_positive", 0, 1), ("false_negative", 1, 0)):
        indices = [i for i, (row, pred) in enumerate(zip(splits.validation, predictions))
                   if row.label == truth and pred == prediction]
        if not indices:
            inspected_errors[name] = None
            print(name, "none in validation")
            continue
        row = splits.validation[indices[0]]
        n_tokens = len(tokenizer(row.text)["input_ids"])
        inspected_errors[name] = {
            "id": row.record_id, "true": row.label, "predicted": prediction,
            "tokens": n_tokens, "truncated": n_tokens > 64,
            "currency_symbol": any(c in row.text for c in "£$€"),
            "digit_count": sum(c.isdigit() for c in row.text),
        }
        print(name, inspected_errors[name])
    print("BERT choice fixed using validation:", selected)
```

In this run the first false positive was `sms-001361`: 109 tokens, so its input was truncated. The first false negative was `sms-000882`: 36 tokens, no currency symbol and no digits. Neither record is explained by a confident claim about model internals. The first motivates a length-policy investigation; the second shows that restoring removed tokens cannot repair every missed spam message.

If a false negative was truncated, inspect the length policy on validation before blaming model capacity. If it was not truncated, increasing the token limit cannot restore evidence that was already present. A false positive matters even when overall accuracy is high: a correspondence workflow may lose a legitimate request. We postpone abstention thresholds to the serving chapters, where quality and review coverage are evaluated together.

## SOLO - make a controlled adaptation change

Implement `adapt_last_layer(model, train_loader, *, encoder_lr=1e-4, epochs=3)`. The supplied model is a freshly loaded two-label BERT; the reusable CPU loader yields dictionaries containing tokenizer tensors and `labels`. Freeze the base, enable only its last layer, keep the classifier trainable, and use AdamW with head rate 0.001, supplied encoder rate and weight decay 0.01. For each epoch select head training/base evaluation mode, clear gradients, run forward/backward/step and return a list of sample-weighted training losses. Mutate the supplied model deliberately; do not edit batches. Valid positive rates and epoch counts are caller obligations. Raise `ValueError` for a loader yielding no samples.

This is the same training skill as the worked recipe with one controlled rate change. Construct a **fresh** model and fresh seed-42 loader; do not continue the already trained winner. Use `score` on validation after the final epoch, compare with the worked final-epoch result, audit frozen/changed tensors, and explain whether doubling the encoder rate helped. Keep three epochs, batch 16, the split and token limit fixed. The final-epoch comparison avoids giving one policy more checkpoint selection opportunities. A worse outcome is acceptable. Test data is not an input to this function.

```python exercise
def adapt_last_layer(model, train_loader, *, encoder_lr=1e-4, epochs=3):
    raise NotImplementedError("Configure gradients and train the final BERT layer plus head")

candidate = new_model("last_layer")
candidate_loader = DataLoader(
    splits.train, batch_size=16, shuffle=True, collate_fn=collate,
    generator=torch.Generator().manual_seed(SEED), num_workers=0)
frozen_before = candidate.bert.embeddings.word_embeddings.weight.detach().clone()
losses = adapt_last_layer(candidate, candidate_loader)
assert len(losses) == 3
assert torch.equal(frozen_before, candidate.bert.embeddings.word_embeddings.weight)
candidate_validation, _ = score(candidate, splits.validation)
print(losses, candidate_validation)
```

Answer and reasoning: `solutions/draft/ch09.py`; tests: `tests/draft/test_ch09.py`. The answer tests include a full real 800-message, three-epoch run when assets are available. A small fake module can test a loop edge case; it cannot establish that the main adaptation exercise ran.

## Score test once, then save the actual inference bundle

The worked recipe and BERT selection are now fixed. Report both frozen BERT candidates and both baselines on test without using this table to revise their settings. Choosing a production family should also use validation and operational constraints. Saving the selected BERT does not assert it beats TF-IDF or has calibrated probabilities.

```python
if ready:
    test_text = [r.text for r in splits.test]
    test_y = [r.label for r in splits.test]
    final_test = {
        "majority": classification_metrics(test_y, [majority] * len(test_y)),
        "tfidf_logistic": classification_metrics(
            test_y, baseline.predict(features.transform(test_text)).tolist()),
    }
    for policy in ("head", "last_layer"):
        final_test[policy], _ = score(models[policy], splits.test)
    print("Final test, after selection:", final_test)
```

**Observed comparison after the worked selection was fixed:**

| Predictor | Validation macro-F1 | Test accuracy | Test macro-F1 | Test spam precision / recall |
| --- | ---: | ---: | ---: | ---: |
| Majority ham | 0.467 | 0.875 | 0.467 | 0.000 / 0.000 |
| TF-IDF + logistic regression | 0.765 | 0.890 | 0.578 | 1.000 / 0.120 |
| BERT head-only, epoch 3 | 0.806 | 0.945 | 0.851 | 0.938 / 0.600 |
| BERT final-layer + head, epoch 2 | 0.912 | 0.980 | 0.953 | 0.957 / 0.880 |

The selected BERT's validation confusion was `[[170,5],[3,22]]`; its final test confusion was `[[174,1],[3,22]]`. Only 25 spam messages occur in each held-out partition: one changed spam prediction moves recall by four percentage points. TF-IDF's perfect spam precision hides its 22 missed test spam messages. This bounded untuned baseline is a useful warning about imbalance, not proof that lexical models are generally inferior. The BERT advantage here does not establish robustness to new sources, and the small test set should temper deployment claims.
An inference bundle needs weights, architecture, label names, tokenizer, length policy and provenance. We save a new directory, record versions and all source/split metadata, then reload both tokenizer and model locally. Token-ID parity alone would miss lost trained weights; logit parity alone on old token tensors would miss a changed tokenizer. We check both.

```python
if ready:
    import json
    import tempfile
    import platform
    import importlib.metadata
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="sms-bert-v1-", dir=ARTIFACTS))
    chosen.eval()
    chosen.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    report = {
        "task_id": "sms-routing.v1", "label_map": {"ham": 0, "spam": 1},
        "base": base_metadata, "split_metadata": splits.metadata,
        "selected": selected, "selection": "validation macro-F1; head wins tie",
        "tokenization": {"max_length": 64, "padding": "longest", "truncation": True},
        "seed": SEED, "batch_size": 16, "epochs": 3, "threads": torch.get_num_threads(),
        "cpu": platform.processor(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "scikit-learn")},
        "runs": runs, "validation": validation, "baseline_validation": baseline_validation,
        "inspected_errors": inspected_errors, "test": final_test,
    }
    (output_dir / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    restored_tokenizer = AutoTokenizer.from_pretrained(
        output_dir, local_files_only=True, trust_remote_code=False)
    restored_model = AutoModelForSequenceClassification.from_pretrained(
        output_dir, local_files_only=True, trust_remote_code=False).eval()
    probe_text = [r.text for r in splits.validation[:8]]
    old_inputs = tokenizer(probe_text, padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
    new_inputs = restored_tokenizer(probe_text, padding=True, truncation=True,
                                   max_length=64, return_tensors="pt")
    for key in old_inputs:
        assert torch.equal(old_inputs[key], new_inputs[key])
    with torch.no_grad():
        expected = chosen(**old_inputs).logits
        actual = restored_model(**new_inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_model.config.id2label == {0: "ham", 1: "spam"}
    print("Model AND tokenizer reload passed:", output_dir.name)
```

The output is a real locally restorable classifier, not a reference to notebook globals or a randomly initialized replacement. The shared later serving model may use the lighter TF-IDF/PyTorch path after comparing costs and quality; BERT adaptation remains an executed capability, and Chapter 22 revisits it with low-rank adapters. Chapter 10 now opens the data preparation boundary that made this comparison credible.

**Completion gate:** train both policies for 150 updates each; verify changed and frozen tensors; compare majority, TF-IDF and BERT with the agreed metrics; inspect available validation error types; independently train the rate variant; explain validation selection and the final test boundary; restore model **and** tokenizer with matching logits. Missing-asset skips leave the central outcome incomplete.

---

## Study 10 — Turn incoming files into a usable dataset

New first-draft study. Source notebook: `book/draft/10-files-and-data-contracts.ipynb`.

# Study 10 — Turn incoming files into a usable dataset

**First draft.** Build a small ingestion boundary that retains good records and reports bad ones. Prerequisites: dictionaries, iteration, and exceptions. This study runs offline with Python's standard library.

## Treat file parsing as a data decision

A model can train successfully on the wrong rows. Ingestion therefore needs an explicit contract before numerical processing starts. A **schema** describes expected fields and types. **Provenance** records where a row came from so a later error can be traced. **Quarantine** holds rejected-row metadata for investigation instead of silently making up replacement values.

Our minimal record has a nonblank string `id` and a string `text`. Blank text is valid at ingestion; a later task may choose to filter it. Category labels are optional here because unlabeled records are useful for inference. For supervised training, validate labels at the training-data boundary.

JSON Lines stores one JSON value per line. It is easy to stream and append, but malformed lines still require a policy. CSV stores a table of string fields; quoting matters when values contain commas or newlines. Neither format validates your business contract automatically.

## Read a tiny file without depending on the working directory

`pathlib.Path` represents filesystem paths. `with` scopes the open file's lifetime, including exception paths. Give text files an explicit encoding rather than relying on a machine default. The example creates a temporary fixture; it does not inspect real customer files.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as folder:
    source_path = Path(folder) / "documents.jsonl"
    fixture = [
        {"id": "a", "text": "Invoice due"},
        {"id": "b", "text": "Claim received"},
    ]
    with source_path.open("w", encoding="utf-8") as handle:
        for record in fixture:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    loaded = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            loaded.append(json.loads(line))
    assert loaded == fixture
    print("Rows loaded:", len(loaded))
```

`enumerate(handle, 1)` pairs each line with a one-based number. We did not use `read()` to load the complete file first. The small `loaded` list is appropriate for a fixture, but a real stream can yield validated records downstream. For very large inputs, also bound record size and define how files are resumed after failure.

## Separate syntax errors from contract errors

Valid JSON can still be an unusable sample: `[]` parses successfully, and `{"id":17}` is a valid object with the wrong fields. Check the parsed value, not just whether parsing raised an exception.

```python
raw_lines = [
    '{"id":"a","text":"Invoice due"}',
    '{broken',
    '{"id":"b","text":17}',
]
usable = []
rejected = []
for line_number, line in enumerate(raw_lines, 1):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        rejected.append({"line": line_number, "reason": "invalid JSON"})
        continue
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        rejected.append({"line": line_number, "reason": "invalid text"})
        continue
    usable.append(record)
assert len(usable) == 1
assert [row["line"] for row in rejected] == [2, 3]
print("Accepted:", len(usable), "Rejected:", rejected)
```

This worked policy is deliberately incomplete: it does not yet validate IDs or duplicates. The exercise finishes that boundary. A rejection report should contain source identifiers, line numbers, and reason codes, not raw sensitive text. Keep the actual source separately under an appropriate retention/access policy.

Catch specific data errors. An I/O failure or programming error should not be transformed into “zero usable rows.” Otherwise the next training stage can appear successful while consuming an empty dataset.

## A CSV boundary uses the same contract

```python
import csv
from io import StringIO

csv_text = 'id,text\na,"Invoice, payment due"\nb,Claim received\n'
rows = list(csv.DictReader(StringIO(csv_text)))
assert rows[0]["text"] == "Invoice, payment due"
assert rows[1]["id"] == "b"
print(rows)
```

`DictReader` uses the header as field names. When opening an actual CSV file, use `open(..., encoding="utf-8", newline="")` so the CSV module controls newline handling. Values remain strings unless you explicitly convert them. A missing column and an empty value are different cases worth testing.

## Essential exercise — one-pass quarantine

Implement `quarantine_jsonl(lines)` returning `(accepted, rejected)`. Ignore blank lines. Accept only objects with a nonblank string ID and string text. Preserve accepted objects unchanged and in order. After the first valid occurrence, quarantine duplicate IDs using exact ID equality. A bad earlier row must not reserve its ID.

Each rejection contains exactly `line` and `reason`, with one-based line numbers. Use reasons `invalid JSON`, `expected object`, `invalid id`, `invalid text`, and `duplicate id`. The iterable supplies strings; source iteration errors propagate.

```python exercise
def quarantine_jsonl(lines):
    raise NotImplementedError("Validate each line and retain a rejection report")

good, bad = quarantine_jsonl([
    '{"id":"a","text":"ok"}', '', '{"id":"a","text":"again"}', '[]'])
assert good == [{"id": "a", "text": "ok"}]
assert bad == [{"line": 3, "reason": "duplicate id"},
               {"line": 4, "reason": "expected object"}]
```

**Optional study:** add a filename and a stable content hash to the report. Decide whether duplicate IDs across files indicate retries, legitimate revisions, or corruption; that decision belongs in an explicit versioning contract.

**Completion gate:** parse JSONL and CSV fixtures, classify malformed syntax versus wrong schema, and demonstrate that a bad row does not silently alter a good one.

Primary sources: https://docs.python.org/3.11/library/json.html and https://docs.python.org/3.11/library/csv.html . Reference answer: `solutions/draft/ch10.py`; tests: `tests/draft/test_ch10.py`.

---

## Study 11 — Join and inspect data before training

New first-draft study. Source notebook: `book/draft/11-pandas-and-sql.ipynb`.

# Study 11 — Join and inspect data before training

**First draft.** Create a small feature table from SQLite and pandas, checking what a join does to row counts. Prerequisites: the ingestion contract and basic Python collections. SQLite is local and serverless; pandas is an optional installed dependency for the second worked section.

## A join can change the experiment

Documents and labels often arrive from different systems. A document table might contain word counts while a reviewer table contains the known category. A **join** combines rows using a shared key. If either side contains duplicate keys, the join can produce more rows than expected. Repeating a sample can change its influence during training.

Before joining, state the expected **cardinality**: one-to-one, one-to-many, or many-to-many. Our fixture expects one label per document ID. We also want to know which documents lack labels instead of quietly treating them as negative examples.

SQLite lets us practice the real SQL boundary without configuring a database server. Parameter binding keeps values separate from SQL syntax. It also avoids quoting errors when a value contains an apostrophe.

## Read a controlled local fixture

```python
import sqlite3

connection = sqlite3.connect(":memory:")
try:
    connection.executescript(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, word_count INTEGER);"
        "CREATE TABLE labels(id TEXT PRIMARY KEY, label TEXT);"
    )
    connection.executemany("INSERT INTO documents VALUES (?, ?)",
                           [("a", 5), ("b", 2), ("c", 9)])
    connection.executemany("INSERT INTO labels VALUES (?, ?)",
                           [("a", "claim"), ("b", "invoice")])
    selected = connection.execute(
        "SELECT id, word_count FROM documents WHERE word_count >= ? ORDER BY id",
        (3,),
    ).fetchall()
    assert selected == [("a", 5), ("c", 9)]
    print("Parameterized query:", selected)
finally:
    connection.close()
```

The tuple `(3,)` contains one bound value; the trailing comma creates a one-item tuple. The `?` placeholder is for a value, not a table or column name. Choose SQL identifiers from trusted code when necessary rather than interpolating arbitrary input.

A primary key enforces uniqueness in this fixture. If your production source has historical versions, an ID alone may not be a unique row key. Select the intended version before joining; `drop_duplicates` without a reason can discard useful information.

## Inspect a pandas join

A pandas **DataFrame** is a table with named columns. Readable inspection comes before feature engineering. Use a left join to retain documents while examining missing labels; then select the labeled rows deliberately.

```python
import importlib.util

if importlib.util.find_spec("pandas") is None:
    print("SKIP pandas join: pandas is not installed.")
else:
    import pandas as pd
    documents = pd.DataFrame({"id": ["a", "b", "c"], "word_count": [5, 2, 9]})
    labels = pd.DataFrame({"id": ["a", "b"], "label": ["claim", "invoice"]})
    inspected = documents.merge(labels, on="id", how="left",
                                 validate="one_to_one", indicator=True)
    missing_ids = inspected.loc[inspected["label"].isna(), "id"].tolist()
    assert missing_ids == ["c"]
    training = inspected.loc[inspected["label"].notna(),
                             ["id", "word_count", "label"]].copy()
    assert training["id"].tolist() == ["a", "b"]
    assert len(training) == 2
    print(training)
```

`validate="one_to_one"` turns an unexpected duplicate-key relationship into an error. `indicator=True` adds a column showing where each output row came from. `.loc[condition, columns]` selects rows and columns by explicit conditions. `.copy()` makes the editable table boundary clear.

A left join does not automatically mean “safe”: inspect null join keys, duplicates, unmatched rows, and source filtering. Pandas can match null keys in ways that differ from usual SQL null semantics. This fixture deliberately uses non-null IDs. Reject or explicitly handle missing identifiers before joining.

Keep split membership visible too. If a customer has several related documents, joining customer metadata does not make those examples independent. Fit numerical transformations using the training partition, and preserve group-based separation when the task requires it.

## Essential exercise — extract the agreed training rows

Implement `joined_training_rows(connection, *, min_words=1)` for the SQLite schema above. Return `(id, word_count, label)` tuples ordered by ID, using an inner join and a bound minimum. Exclude documents without labels and rows below the minimum. Reject a negative minimum with `ValueError`; the caller supplies an integer. Leave the supplied connection open.

```python exercise
def joined_training_rows(connection, *, min_words=1):
    raise NotImplementedError("Join by ID and bind the filter value")

con = sqlite3.connect(":memory:")
try:
    con.executescript(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, word_count INTEGER);"
        "CREATE TABLE labels(id TEXT PRIMARY KEY, label TEXT);")
    con.execute("INSERT INTO documents VALUES (?, ?)", ("a", 5))
    con.execute("INSERT INTO labels VALUES (?, ?)", ("a", "claim"))
    assert joined_training_rows(con, min_words=3) == [("a", 5, "claim")]
finally:
    con.close()
```

**Optional study:** insert a duplicate label into a pandas fixture and verify the cardinality check fails. Compare the row count with an unchecked many-to-many merge.

**Completion gate:** explain the query placeholders, identify the unlabeled document, and demonstrate that filtering and joining preserve the intended training population.

Primary sources: https://docs.python.org/3.11/library/sqlite3.html and https://pandas.pydata.org/docs/user_guide/merging.html . Reference answer: `solutions/draft/ch11.py`; tests: `tests/draft/test_ch11.py`.

---

## Study 12 — Version the data transformation

New first-draft study. Source notebook: `book/draft/12-dataset-artifacts.ipynb`.

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

---

## Study 13 — Search by vectors and evaluate the neighbors

New first-draft study. Source notebook: `book/draft/13-embeddings-and-retrieval.ipynb`.

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

---

## Study 14 — Build an evidence-first RAG workflow

New first-draft study. Source notebook: `book/draft/14-rag-and-grounding.ipynb`.

# Study 14 — Build an evidence-first RAG workflow

**First draft.** Connect chunking, retrieval, source citations, and abstention in an offline workflow. Prerequisites: vector retrieval and relevance fixtures. The response stage here is a deterministic extractive baseline, **not an LLM**. Its limitations are part of the study.

## Retrieval and generation solve different problems

**Retrieval-augmented generation (RAG)** supplies retrieved evidence to a generator before it answers. Retrieval chooses candidate material; generation decides how to use it. A system can retrieve poorly and sound fluent, or retrieve the right material and still produce an unsupported answer.

Start with inspectable intermediate results. Each source document gets a stable ID. **Chunking** breaks it into smaller passages while retaining source references. Overlap can preserve context across boundaries, but it creates duplicate information and additional index entries. Choose chunk size based on the downstream representation and task rather than assuming a universal number.

For this small lab, chunk by whitespace words. Model token counts can differ, so this is not a substitute for measuring the actual tokenizer's input length.

## Keep provenance while chunking

```python
sources = {
    "billing": "Invoice payments are due within thirty days. Contact billing for a receipt.",
    "claims": "Vehicle claims need a repair estimate. Submit the estimate with your claim.",
}

def chunk_words(source_id, text, *, size=7, overlap=2):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > overlap >= 0")
    words = text.split()
    chunks = []
    for start in range(0, len(words), size - overlap):
        chunks.append({"id": f"{source_id}:{start}", "source": source_id,
                       "text": " ".join(words[start:start + size])})
        if start + size >= len(words):
            break
    return chunks

chunks = []
for source_id, text in sources.items():
    chunks.extend(chunk_words(source_id, text))
assert all(chunk["source"] in sources for chunk in chunks)
assert len({chunk["id"] for chunk in chunks}) == len(chunks)
print(chunks[0])
```

The chunk ID includes its source and starting word offset. If a document changes, a production system also needs a source version; otherwise the same ID might refer to different text after reindexing. Keep the chunk text and source version together.

## Retrieve before answering

To keep the response behavior transparent, rank by token overlap rather than introducing another model. This lexical method misses synonyms and can overvalue generic words. The goal is a visible contract, not competitive search quality.

```python
def words(text):
    return set(text.lower().strip(".?!").split())

def retrieve(question, passages, *, k=2):
    query_words = words(question)
    scored = [(len(query_words & words(p["text"])), p) for p in passages]
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [passage for score, passage in scored[:k] if score > 0]

evidence = retrieve("invoice payments", chunks)
assert evidence
assert evidence[0]["source"] == "billing"
assert retrieve("penguin habitat", chunks) == []
print("Retrieved IDs:", [item["id"] for item in evidence])
```

The filtering policy refuses zero-overlap passages. That is better than returning an arbitrary first chunk, but a positive overlap is still not proof that the passage answers the question. A mature system needs stronger retrieval evaluation and calibrated answer policies.

## An extractive response makes support visible

```python
def extractive_response(question, passages):
    evidence = retrieve(question, passages)
    if not evidence:
        return {"answer": None, "citations": [], "status": "abstain"}
    selected = evidence[0]
    return {"answer": selected["text"], "citations": [selected["id"]],
            "status": "extractive"}

response = extractive_response("invoice payments", chunks)
assert response["status"] == "extractive"
cited = next(p for p in chunks if p["id"] == response["citations"][0])
assert response["answer"] == cited["text"]
assert extractive_response("penguin habitat", chunks)["status"] == "abstain"
print(response)
```

This copies evidence verbatim. It is easy to verify the citation, but it does not synthesize an answer or guarantee relevance. If you replace it with an LLM, pass bounded evidence, preserve citation IDs, and treat retrieved text as data rather than trusted instructions. A source saying “ignore the system rules” must not become an instruction to the application.

Separate evaluation questions: did retrieval include the known relevant source? Did the response use only supported facts? Did the system abstain when evidence was absent? Do not collapse those questions into a single impression of answer quality.

## Essential exercise — measure retrieval coverage

Implement `recall_at_k(retrieved_ids, relevant_ids, k)`. Divide the number of unique relevant IDs recovered in the first `k` positions by the number of unique relevant IDs. Duplicate retrieved IDs receive no extra credit. Reject an empty relevance set or nonpositive `k` with `ValueError`. The caller supplies ID sequences and an integer. Empty retrieved results return zero.

```python exercise
def recall_at_k(retrieved_ids, relevant_ids, k):
    raise NotImplementedError("Count recovered relevant IDs in the first k positions")

assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 0.5
assert recall_at_k(["wrong", "b", "a"], ["a", "b"], 3) == 1.0
assert recall_at_k([], ["a"], 2) == 0.0
```

**Optional study:** replace the extractive baseline with a local generator. Keep the retrieval fixture unchanged, add unsupported-answer and no-evidence cases, and report generation latency separately. A fluent answer is not itself evidence of grounding.

**Completion gate:** retain source IDs through chunking, demonstrate a supported extract and an abstention, then calculate retrieval recall independently of response wording.

Primary implementation study: https://huggingface.co/learn/cookbook/en/advanced_rag . Reference answer: `solutions/draft/ch14.py`; tests: `tests/draft/test_ch14.py`. No hosted model or actual LLM generation is executed here.

---

## Study 15 — Run inference with measurable tradeoffs

New first-draft study. Source notebook: `book/draft/15-inference-performance.ipynb`.

# Study 15 — Inference That You Can Measure

**First draft. Core study, about 40 minutes.** You already have a model and held-out evaluation. Now the task is to turn a prediction function into a predictable batch operation and measure the part of the system that actually matters. The deliverable is an ordered, bounded inference adapter, with tests for the final partial batch.

## Decide what a measurement represents

**Latency** is elapsed time for an operation, such as responding to one request. **Throughput** counts completed work per unit time. A batch can improve throughput while making an individual request wait longer. Neither metric, alone, tells you whether the service meets its goal.

Start with a deliberately simple predictor so batching and timing remain visible. It doubles a numerical feature; it is not a trained classifier. `perf_counter` measures elapsed time with a clock suitable for short durations. Keep the result so the example checks useful work, not just a loop that happens to execute.

```python
from time import perf_counter

def predict_features(rows):
    return [row * 2 for row in rows]

rows = list(range(1000))
started = perf_counter()
predictions = predict_features(rows)
elapsed = perf_counter() - started
assert predictions[:3] == [0, 2, 4]
assert len(predictions) == len(rows)
print({"items": len(rows), "elapsed_seconds": elapsed})
```

Your number is an observation on your machine, not a performance claim for another machine or model. Tiny examples include interpreter, allocation, and timer overhead. Repeat a realistic workload, report distributions rather than one favorable run, and separate startup/model-loading time from steady-state prediction. Do not put “must run in less than one millisecond” into a portable unit test.

## Use the right model mode

PyTorch has two distinct controls. `model.eval()` changes the behavior of layers such as dropout; `torch.inference_mode()` disables gradient-related work for inference. Evaluation mode does not itself disable gradient tracking. The following optional dependency block runs real CPU tensor operations when PyTorch is installed; it downloads nothing and does not train a model.

```python
try:
    import torch
except ModuleNotFoundError:
    print("SKIP PyTorch mode demonstration: install the book's CPU environment.")
else:
    torch.manual_seed(15)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(0.5))
    model.eval()
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with torch.inference_mode():
        first = model(features)
        second = model(features)
    assert torch.equal(first, second)
    assert not first.requires_grad
    print("Inference output shape:", tuple(first.shape))
```

On a GPU, host-side timing can finish before asynchronous device work completes. Use appropriate device synchronization or PyTorch benchmarking tools when you later measure accelerators. This study stays on CPU and makes no CPU/GPU comparison.

## Keep batches bounded and answers aligned

A service rarely receives an exactly convenient number of samples. Five records and a batch size of two require calls of sizes two, two, and one. Dropping the last record is a correctness defect, even if the larger batches look faster.

The input might be a one-shot generator from a file reader. Accumulate one small list, call the predictor, append its returned results, and replace that small list. You need memory for the current batch and, for this return contract, all output predictions. A streaming-result API could avoid retaining every result, but it would be a different API with different failure behavior.

### SOLO exercise — implement the inference adapter

Implement `batched_predict(values, predictor, batch_size=8)`. Accept an iterable and a callable returning one result per input item. Return a new flat list preserving input order. Process at most `batch_size` inputs per call, keep the tail, and do not call the predictor for empty input. Reject non-integer sizes, including Boolean values, with `TypeError`; reject nonpositive sizes or a predictor result of the wrong length with `ValueError`. Inputs are consumed once. A predictor error may happen after earlier batches were processed; no rollback is promised.

```python exercise
def batched_predict(values, predictor, batch_size=8):
    raise NotImplementedError("Implement bounded inference batching")

assert batched_predict(iter(range(5)), lambda b: [x * 2 for x in b], 2) == [0, 2, 4, 6, 8]
assert batched_predict([], lambda b: b, 2) == []
```

**Completion gate:** demonstrate the tail, empty input, result alignment, and a bad predictor-result count. Explain why an output-equivalence assertion belongs in a benchmark. **Optional study:** measure several batch sizes on your actual trained model, recording both throughput and per-request latency; do not assume the largest batch wins.

Reference answers: `solutions/draft/ch15.py`; tests: `tests/draft/test_ch15.py`. Primary sources: [Python timing](https://docs.python.org/3.11/library/time.html#time.perf_counter), [PyTorch inference mode](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html), and [PyTorch benchmarking](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html). Next, keep this prediction contract intact when adding an HTTP boundary.

---

## Study 16 — Expose the model through a small service

New first-draft study. Source notebook: `book/draft/16-model-serving.ipynb`.

# Study 16 — Put a Stable Boundary Around a Model

**First draft. Core study, about 40 minutes.** A notebook function becomes useful to another application when it has an explicit request and response contract. The goal here is a small prediction boundary that you can test without starting a server. HTTP is an adapter around that boundary, not the place to hide preprocessing or model logic.

## Separate application behavior from transport

An inference service needs to know which fields are allowed, what a valid text looks like, and which labels can be returned. A function taking a dictionary is enough to practice those choices. The predictor below is a transparent keyword rule so that service behavior is testable without model files. Replace it with your loaded model later; do not mistake its outputs for an evaluated ML result.

```python
def keyword_predictor(text):
    words = text.lower().split()
    if "invoice" in words:
        return "invoice"
    if "claim" in words:
        return "claim"
    return "general"

assert keyword_predictor("Invoice due") == "invoice"
assert keyword_predictor("Please review my claim") == "claim"
```

An application function should reject a bad request before invoking the expensive predictor. Here a valid request has exactly one key, `text`; its value is a string, contains a non-whitespace character, and has at most 200 characters. A size bound is a resource contract, not a comprehensive abuse defense. A tokenizer-based production service may also limit tokens and total batch size.

```python
def handle_prediction(payload):
    if not isinstance(payload, dict):
        raise TypeError("request must be a dictionary")
    if set(payload) != {"text"}:
        raise ValueError("request must contain exactly text")
    text = payload["text"]
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip() or len(text) > 200:
        raise ValueError("invalid text length")
    return {"label": keyword_predictor(text)}

request = {"text": "Invoice due"}
assert handle_prediction(request) == {"label": "invoice"}
assert request == {"text": "Invoice due"}
print(handle_prediction(request))
```

`set(payload)` visits dictionary keys, so comparison with `{"text"}` rejects both missing and extra fields. Explicitly validating the input means an accidental `authorization` or `customer_id` field does not silently enter model processing. It does not mean the application should echo rejected values into logs or errors.

## Add FastAPI as an adapter

FastAPI uses Python type declarations and Pydantic models to interpret request bodies. This complete optional example constructs an app but starts no listener. Save the app in a module if you later choose to run an ASGI server. Core validation above remains runnable without FastAPI.

```python
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, ConfigDict
except ModuleNotFoundError:
    print("SKIP HTTP adapter: optional FastAPI/Pydantic dependencies are absent.")
else:
    class PredictionRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        text: str = Field(min_length=1, max_length=200)

    app = FastAPI()

    @app.post("/predict")
    def predict_endpoint(body: PredictionRequest):
        try:
            return handle_prediction({"text": body.text})
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail="Invalid prediction request") from error

    assert predict_endpoint(PredictionRequest(text="Invoice due")) == {"label": "invoice"}
    print("App constructed without starting a server.")
```

This is an application adapter, not a production deployment. A deployed service also needs trusted model loading, model/version readiness, authentication, resource limits, timeouts, error handling, and operational monitoring. Load a trusted model once per worker rather than rebuilding it for each request. Avoid doing CPU-heavy inference directly in an asynchronous event loop just because `async def` is available.

### SOLO exercise — inject the actual predictor

Implement `predict_request(payload, predictor, *, max_chars=200)`. Follow the dictionary/text contract above, but call the supplied predictor. The caller promises a positive integer `max_chars`. Verify that its result is one of `claim`, `invoice`, or `general`; otherwise raise `ValueError`. Return exactly `{"label": result}` without editing the request. Reject invalid request types with `TypeError` and invalid fields/lengths with `ValueError`.

```python exercise
def predict_request(payload, predictor, *, max_chars=200):
    raise NotImplementedError("Implement a tested prediction boundary")

assert predict_request({"text": "hello"}, lambda text: "general") == {"label": "general"}
```

**Completion gate:** prove malformed input does not call the predictor, valid input is preserved, and an unknown model label fails visibly. **Optional study:** use FastAPI's `TestClient` with optional HTTPX installed to assert HTTP status codes and response bodies in-process. A direct endpoint-function call does not test JSON parsing or middleware, so record that distinction in your validation evidence.

Reference answers: `solutions/draft/ch16.py`; tests: `tests/draft/test_ch16.py`. Primary sources: [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/) and [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/). Next, persist the workflow state surrounding a prediction so a process can restart without forgetting completed work.

---

## Study 17 — Compose restartable AI workflow steps

New first-draft study. Source notebook: `book/draft/17-workflow-orchestration.ipynb`.

# Study 17 — A Workflow That Remembers Its Work

**First draft. Core study, about 45 minutes.** A document workflow can validate a record, prepare features, invoke a model, and save the result. A process crash should not erase every completed decision. This study introduces durable state with SQLite and uses an explicit tool registry instead of arbitrary executable instructions.

## State belongs outside the worker process

A Python dictionary is a useful cache, but it disappears when the process exits. A durable **checkpoint** stores enough information to continue after restart. A job ID identifies a logical piece of work; a completed result associated with that ID can be reused when the caller retries.

Use an actual temporary database file rather than an in-memory database for the restart experiment. Closing and reopening a connection demonstrates persistence without a database server. A `TemporaryDirectory` owns the example's files and cleans them up afterward.

```python
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as folder:
    database = Path(folder) / "workflow.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("CREATE TABLE results (job_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute("INSERT INTO results VALUES (?, ?)",
                               ("doc-1", json.dumps({"label": "invoice"})))
    with closing(sqlite3.connect(database)) as reopened:
        row = reopened.execute("SELECT payload FROM results WHERE job_id = ?", ("doc-1",)).fetchone()
        assert json.loads(row[0]) == {"label": "invoice"}
        print("Recovered:", json.loads(row[0]))
```

The inner connection context commits on normal exit and rolls back on an exception. It does not close the connection; `closing(...)` handles that separate resource lifecycle. The question marks are parameter placeholders. Values are supplied separately, rather than concatenated into SQL, so an ID cannot turn into SQL syntax.

Our table stores a JSON result, not Python pickle data or arbitrary executable objects. JSON has a limited data model: dictionaries, lists, strings, numbers, Booleans, and null. Define a result schema before integrating multiple independent workers.

## Select tools through an explicit contract

An orchestrator chooses which operation runs next. When an LLM later suggests a tool name, that suggestion is still data. Resolve it through a trusted registry; never evaluate generated Python or shell text.

```python
def word_count(text):
    return {"word_count": len(text.split())}

tools = {"count_words": word_count}

def dispatch(name, text):
    if name not in tools:
        raise ValueError("unknown tool")
    if not isinstance(text, str) or len(text) > 1000:
        raise ValueError("tool input must be bounded text")
    return tools[name](text)

assert dispatch("count_words", "small reliable workflow") == {"word_count": 3}
print(dispatch("count_words", "Invoice due"))
```

This registry defines capabilities: the orchestrator can count words, not read arbitrary files or execute commands. A production tool would also need authorization for its data and effects. A correct tool name does not establish permission to perform a particular action.

### SOLO exercise — reuse a completed result after restart

Implement `run_once(database, job_id, handler)`. For this single-worker lab, open SQLite, create a `jobs(id TEXT PRIMARY KEY, result TEXT NOT NULL)` table if needed, and return the stored JSON dictionary if `job_id` is present. Otherwise call `handler()`, require a JSON-serializable dictionary, save it, and return its JSON-round-tripped value. Reject a blank/non-string ID with `ValueError`; a non-dictionary result raises `TypeError`. Propagate handler/serialization/database errors, and do not record a successful result when they happen. Use `json.dumps(..., allow_nan=False)` and parameterized SQL.

```python exercise
def run_once(database, job_id, handler):
    raise NotImplementedError("Implement durable result reuse")

# In your own temporary directory, call twice using the same database and ID.
# The second handler should raise if invoked; the stored result should be returned instead.
```

The function name describes this lab's normal retry behavior, not a global exactly-once guarantee. A worker can crash after an external side effect but before storing the result. The retry may perform that effect again. Likewise, two workers can both see an absent result unless you introduce a proper claim/lease protocol. Do not apply this simple single-worker pattern to concurrent payment, email, or mutation tools without stronger coordination and idempotency support.

**Completion gate:** show persistence across closed connections, a failed handler followed by a successful retry, and an ID containing quote characters. Explain the crash window in your own words. **Optional study:** add explicit `pending`, `running`, `completed`, and `failed` states, with attempt counts and a lease expiry, then reason about recovery before implementing parallel workers.

Reference answers: `solutions/draft/ch17.py`; tests: `tests/draft/test_ch17.py`. Primary source: [Python SQLite transactions, parameters, and connection management](https://docs.python.org/3.11/library/sqlite3.html). Next, record operational evidence without copying private document contents into telemetry.

---

## Study 18 — Make failures visible without leaking data

New first-draft study. Source notebook: `book/draft/18-observability-and-safety.ipynb`.

# Study 18 — Observe Decisions Without Logging Documents

**First draft. Core study, about 40 minutes.** A prediction service needs enough evidence to investigate failures: which model ran, how long it took, whether validation failed, and which trace connects its stages. Logging the entire document is an easy shortcut that can create a second, poorly controlled copy of sensitive data.

The deliverable is a small, allowlisted audit event and an operational summary. The same principle applies to API tokens, retrieved passages, prompts, customer metadata, and tool responses: useful telemetry is a designed data product, not a dump of every object in memory.

## Start with questions the event must answer

A **trace ID** connects events belonging to one request. A **model version** identifies the artifact/configuration used for prediction. Latency describes elapsed time, while a small decision vocabulary makes failures countable. These fields support operational investigation without needing the original text in the event.

Use a generated identifier, not an email address or customer number. Even a schema containing only `request_id` can carry personal information if the caller fills it with personal information. Field names are not a privacy guarantee.

```python
import json
from uuid import uuid4

source = {
    "request_id": str(uuid4()), "model_version": "local-classifier-v1",
    "latency_ms": 12.5, "decision": "accepted",
    "text": "A private document", "authorization": "do-not-log-this",
}
operational = {name: source[name] for name in
               ("request_id", "model_version", "latency_ms", "decision")}
encoded = json.dumps(operational, sort_keys=True)
assert "private" not in encoded
assert "do-not-log-this" not in encoded
print(encoded)
```

The dictionary comprehension selects approved fields. This is more stable than a blacklist of current secret names: a newly added payload field is excluded by default. Keep the source record unchanged so validation and model execution see the original request.

## Turn events into useful operational checks

Counts and ratios can reveal problems before someone inspects individual logs. A rising rejected-request count may indicate a schema change, bad upstream data, or misuse. It does not, by itself, show degraded model accuracy. Model quality needs evaluation against trustworthy labels, sometimes collected later.

```python
events = [
    {"decision": "accepted", "latency_ms": 8.0},
    {"decision": "rejected", "latency_ms": 1.0},
    {"decision": "accepted", "latency_ms": 10.0},
]
counts = {}
for event in events:
    decision = event["decision"]
    counts[decision] = counts.get(decision, 0) + 1
assert counts == {"accepted": 2, "rejected": 1}
print("Decision counts:", counts)
```

A drift signal describes a change in the inputs or predictions over time. It is a reason to investigate, not automatic proof that a new model is needed. Compare like periods and data sources, preserve an evaluation baseline, and distinguish changes in traffic mix from actual error changes.

Python logging can carry the serialized event to the application's configured destination. Libraries should obtain a named logger and leave output configuration to the application. Do not log entire caught exceptions blindly: error messages may contain user-provided text or credentials.

### SOLO exercise — validate and filter an audit event

Write `audit_event(event)` returning a new dictionary with exactly `request_id`, `model_version`, `latency_ms`, and `decision`. Required keys missing from the dictionary raise `KeyError`. IDs/version must be nonblank strings, latency must be a finite, nonnegative integer or float (reject Boolean values with `TypeError`), and decision must be `accepted`, `rejected`, or `failed`. Invalid values raise `ValueError`. Convert latency to float, ignore all other fields, and never mutate the source. The caller owns the promise that its identifiers contain no sensitive information.

```python exercise
def audit_event(event):
    raise NotImplementedError("Create a validated allowlisted event")

event = {"request_id": "trace-1", "model_version": "v1", "latency_ms": 3,
         "decision": "accepted", "text": "private"}
assert audit_event(event) == {"request_id": "trace-1", "model_version": "v1",
                              "latency_ms": 3.0, "decision": "accepted"}
assert event["text"] == "private"
```

Use `math.isfinite` rather than checking only `latency >= 0`: NaN and infinity are valid floating-point values but poor operational measurements. The allowlist removes raw content from this event; it does not remove that content from every log, tracing span, exception handler, or third-party monitoring agent. Review the complete data path, access controls, and retention policy when deploying the system.

**Completion gate:** prove an unexpected secret field is excluded, an invalid latency is rejected, and the source survives unchanged. Explain the difference between a service-health metric and a model-quality metric. **Optional study:** add a windowed rejection-rate alert with a minimum sample count; explain how a very small traffic window could otherwise create noisy alerts.

Reference answers: `solutions/draft/ch18.py`; tests: `tests/draft/test_ch18.py`. Primary sources: [Python logging](https://docs.python.org/3.11/library/logging.html) and [finite-number checks](https://docs.python.org/3.11/library/math.html#math.isfinite). The next optional studies return to model internals: image/sequence shapes and attention.

---

## Study 19 — Match architecture to input structure

New first-draft study. Source notebook: `book/draft/19-vision-and-sequences.ipynb`.

# Study 19 — Images and Sequences Change the Shape Contract

**First draft. Optional architecture study, about 45 minutes.** You can train and fine-tune a pretrained model before completing this study. Return here when image or sequence structure affects the model you choose. The goal is to trace a small CNN and recurrent encoder, then prevent padded sequence positions from corrupting a pooled representation. All code uses small CPU tensors and downloads nothing.

## An image is more than a flattened row

A convolutional neural network (**CNN**) applies learned filters across spatial positions. Sharing a filter lets the same local pattern detector operate in different regions. This gives a useful structural assumption for images, but does not guarantee that a trained model will recognize every shift, rotation, or document layout.

PyTorch's two-dimensional convolution expects a batch with axes `[batch, channels, height, width]`. A grayscale image has one channel; an RGB image normally has three. The network below uses synthetic images to verify shape flow. Randomly initialized filters are not a trained vision system.

```python
import torch
from torch import nn

torch.manual_seed(19)
images = torch.randn(2, 1, 8, 8)
convolution = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
activation_maps = torch.relu(convolution(images))
assert tuple(activation_maps.shape) == (2, 4, 6, 6)
pooled_images = activation_maps.mean(dim=(2, 3))
classifier = nn.Linear(4, 3)
scores = classifier(pooled_images)
assert tuple(scores.shape) == (2, 3)
print("Images -> maps -> pooled -> class scores:",
      images.shape, activation_maps.shape, pooled_images.shape, scores.shape)
```

With no padding and stride one, a width-three filter can occupy six positions along an eight-element axis. Four learned filters produce four output channels. Taking the mean over the height and width axes leaves one number per channel. A linear head maps those four numbers to three class scores. Labels would have shape `[2]` for the class-index loss contract used earlier.

Do not confuse this manual pooling with a universal best design. For page images, resizing, aspect ratios, text resolution, augmentation, and a pretrained backbone may matter more than the toy architecture choice. Always evaluate on representative documents rather than synthetic shape fixtures.

## A sequence carries order

A recurrent layer processes positions in sequence and carries a hidden state. A gated recurrent unit (**GRU**) learns how to update that state. With `batch_first=True`, its input layout is `[batch, time, features]`. Here each of three positions has four input features; these could be embeddings or sensor measurements.

```python
sequence = torch.randn(2, 3, 4)
encoder = nn.GRU(input_size=4, hidden_size=5, batch_first=True)
outputs, final_state = encoder(sequence)
assert tuple(outputs.shape) == (2, 3, 5)
assert tuple(final_state.shape) == (1, 2, 5)
print("Per-position representations:", outputs.shape)
```

`outputs` contains a hidden representation at every time position. The leading axis of `final_state` counts recurrent layers and directions, not batch size. A single one-direction layer gives size one there. LSTM layers add a separate cell state; do not substitute their return structure without checking the API contract.

## Padding is storage, not evidence

Real sequences have different lengths. A batch may pad the shorter ones to a common width. Those extra positions should not influence a pooled feature merely because they occupy storage. Even padding with zero changes an unmasked average: dividing by the padded width shrinks a short sequence's representation.

```python
representations = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]])
valid_length = 2
expected = representations[0, :valid_length].mean(dim=0)
assert torch.equal(expected, torch.tensor([2.0, 4.0]))
assert not torch.allclose(representations.mean(dim=1)[0], expected)
print("The padded value must not contribute:", expected)
```

This checks pooling, not recurrent padding behavior. A recurrent layer can also update its state while consuming padded positions. Packed sequences or a suitable length-aware strategy are separate choices when building an actual recurrent classifier.

### SOLO exercise — length-aware pooling

Write `masked_mean(batch, lengths)` for floating-point input `[B,T,D]` and integer lengths `[B]`. Return `[B,D]`, averaging only positions before each length. Reject incompatible ranks/batch sizes or lengths outside `1..T` with `ValueError`; reject nonfloating input or noninteger lengths with `TypeError`. Do not change the input. The caller supplies finite values and compatible devices; moving lengths to the batch device is permitted. A Boolean mask from `torch.arange(T)` and broadcasting can mark valid positions.

```python exercise
def masked_mean(batch, lengths):
    raise NotImplementedError("Pool only genuine sequence positions")

values = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
assert torch.equal(masked_mean(values, torch.tensor([2])), torch.tensor([[2., 4.]]))
```

**Completion gate:** explain every shape above and prove changing a padded value cannot change the pooled result. **Optional extension:** verify that padded positions receive zero gradient through your pooler, then compare a packed GRU with an unpacked, padded GRU on a tiny fixture.

Reference answers: `solutions/draft/ch19.py`; tests: `tests/draft/test_ch19.py`. Primary sources: [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html), and [packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html). Next, study a different way to mix information across sequence positions: attention.

---

## Study 20 — See what attention computes

New first-draft study. Source notebook: `book/draft/20-attention.ipynb`.

# Study 20 — Attention as a Tested Information-Flow Operation

**First draft. Optional architecture study, about 45 minutes.** A pretrained transformer can be useful before you implement attention. This study opens one small part of that machinery so you can reason about shapes, masks, and information leakage. The deliverable is one causal attention head with tests proving that future values cannot affect an earlier output. It is not a complete transformer.

## Match, normalize, combine

Attention constructs each output by combining values from other positions. A **query** asks what information a position wants; a **key** describes information available at a position; a **value** supplies the representation to combine. Learned projections normally construct these tensors from input representations. We supply them directly to keep the operation visible.

With query and key shape `[T,D]`, the matrix product `query @ key.T` has shape `[T,T]`. Entry `[i,j]` measures how strongly position `i` matches position `j`. Divide by the square root of `D`, then apply softmax across the key positions. Softmax produces nonnegative weights summing to one in each row. Multiplying those weights by values `[T,Dv]` gives an output `[T,Dv]`.

```python
import math
import torch

query = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
key = query.clone()
value = torch.tensor([[10., 0.], [0., 10.], [100., 100.]])
scores = query @ key.T / math.sqrt(query.shape[-1])
weights = torch.softmax(scores, dim=-1)
mixed = weights @ value
assert scores.shape == (3, 3)
assert mixed.shape == (3, 2)
assert torch.allclose(weights.sum(dim=-1), torch.ones(3))
print("Unmasked attention weights:\n", weights)
```

The scaling moderates the size of dot-product scores as feature width grows. It does not replace training, normalization, or numerical checks for an actual model. Our finite float inputs keep the arithmetic straightforward; mixed precision and very long sequences need additional care.

## A causal mask controls permitted information

An autoregressive language model predicts the next token using earlier context. During training, placing all tokens in one matrix is efficient, but position zero must not read token two's information. A **causal mask** allows the current position and earlier ones, and blocks later positions.

```python
allowed = torch.ones(3, 3, dtype=torch.bool).tril()
masked_scores = scores.masked_fill(~allowed, -torch.inf)
causal_weights = torch.softmax(masked_scores, dim=-1)
causal_output = causal_weights @ value
assert torch.equal(causal_weights.triu(1), torch.zeros(3, 3))
assert torch.allclose(causal_output[0], value[0])
print("Causal weights:\n", causal_weights)
```

`tril()` keeps the lower triangle, including the diagonal. `~allowed` inverts the Boolean mask, selecting forbidden positions. Their negative-infinity scores become zero softmax weight. Every row retains at least its diagonal position, so no row is entirely blocked. An entirely blocked row would need explicit handling rather than relying on softmax to produce a sensible result.

Mask conventions differ across APIs. Some APIs interpret a Boolean `True` as allowed, while others use it to mark blocked positions. Read the actual function's contract; copying a Boolean mask without checking its meaning can silently reverse the intended access rule.

## Test causality through a counterfactual

A zero upper triangle is useful, but a behavioral test is stronger: change a future value and confirm that earlier outputs stay fixed. This is the same engineering habit used to test data isolation in preprocessing.

```python
changed_value = value.clone()
changed_value[2] = torch.tensor([-500., 500.])
changed_output = causal_weights @ changed_value
assert torch.allclose(changed_output[:2], causal_output[:2])
assert not torch.allclose(changed_output[2], causal_output[2])
print("Future value changes leave earlier outputs unchanged.")
```

This test isolates the value-combination path. If a larger model computes keys or earlier hidden representations using future information before attention, the mask alone cannot repair that earlier leakage. Inspect the complete data path when reasoning about causality.

### SOLO exercise — package the attention head

Write `causal_attention(query, key, value)` returning `(output, weights)`. Accept finite floating-point rank-two tensors on compatible devices/dtypes. Query and key must have the same nonempty `[T,D]` shape; values must have the same token count and may have a different feature width. Raise `ValueError` for invalid ranks/shapes or empty query dimensions, and `TypeError` for nonfloating inputs. Do not modify the inputs. Use scaled dot-product scores, a causal mask, and row-wise softmax.

```python exercise
def causal_attention(query, key, value):
    raise NotImplementedError("Build a causal attention head")

q = torch.eye(3)
out, attention = causal_attention(q, q, torch.arange(6.).reshape(3, 2))
assert out.shape == (3, 2)
assert torch.equal(attention.triu(1), torch.zeros(3, 3))
assert torch.allclose(attention.sum(-1), torch.ones(3))
```

**Completion gate:** trace the shapes, test row normalization, and reproduce the future-value counterfactual with your function. **Optional study:** compare outputs with PyTorch's `scaled_dot_product_attention` using `is_causal=True` and `dropout_p=0.0`. Then add separate batch/head axes and explain where memory grows quadratically with sequence length. Multi-head projections, residual connections, normalization, and feed-forward layers belong to the next architectural step.

Reference answers: `solutions/draft/ch20.py`; tests: `tests/draft/test_ch20.py`. Primary source: [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html). This study provides an inspectable correctness model, not a performance replacement for optimized framework attention kernels.

---

## Study 21 — Train a tiny language-model experiment

New first-draft study. Source notebook: `book/draft/21-tiny-language-model.ipynb`.

# Study 21 — Train a tiny next-character model

**First draft; optional architecture study.** You have already trained a classifier and used pretrained models. Now make the language-model objective visible in a system small enough to inspect. This experiment uses CPU PyTorch and local text; it downloads nothing.

A language model assigns probabilities to possible continuations. In next-token training, the target at each position is the token immediately after the input context. Here a token is one character and the context is just the current character. The result is a **bigram model**, not a transformer or a useful assistant. This deliberately limited model separates the training objective from sophisticated architecture.

## Construct the prediction task

Consider `aba`: its supervised examples are `a → b` and `b → a`. We do not give the model `a → a` at the same position, because that would teach copying instead of continuation. Python slicing expresses the shift directly. The vocabulary maps characters to integer IDs; those IDs are identifiers, not numerical measurements.

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
text = "ab ab ab ab ab ab "
characters = sorted(set(text))
stoi = {character: index for index, character in enumerate(characters)}
ids = torch.tensor([stoi[character] for character in text], dtype=torch.long)
x, y = ids[:-1], ids[1:]
assert len(x) == len(y) == len(text) - 1
assert text[1] == characters[y[0].item()]
print("Vocabulary:", stoi, "Training pairs:", len(x))
```

Each output must contain one score per vocabulary item. `nn.Embedding(vocabulary_size, vocabulary_size)` is a learned lookup table: looking up the current character returns its scores for the next character. An embedding usually represents tokens as shorter vectors, but nothing prevents its output width from equaling the vocabulary size here. Cross-entropy compares raw scores with integer target IDs; applying softmax before that loss would be redundant and numerically undesirable.

```python
model = nn.Embedding(len(characters), len(characters))
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
with torch.no_grad():
    initial_loss = F.cross_entropy(model(x), y).item()
for step in range(40):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    final_loss = F.cross_entropy(model(x), y).item()
assert final_loss < initial_loss
print("Observed training loss:", initial_loss, "->", final_loss)
```

The assertion checks that this bounded optimization learned something about its training corpus. It does not demonstrate generalization. Repeated training on one tiny string can memorize transitions; there is no independent test corpus here. The exact printed losses are observations from your environment, not a performance benchmark.

## Generate from the learned distribution

Generation repeatedly predicts, selects a continuation, appends it, and uses that character as the next input. **Temperature** rescales scores before softmax. Lower positive values concentrate probability on high-scoring characters; higher values spread it out. Sampling uses a separately seeded generator so you can repeat the experiment without resetting model initialization.

```python
generator = torch.Generator().manual_seed(11)
current = torch.tensor([stoi["a"]])
output = ["a"]
with torch.no_grad():
    for _ in range(12):
        probabilities = torch.softmax(model(current) / 0.7, dim=-1)
        current = torch.multinomial(probabilities, 1, generator=generator).reshape(1)
        output.append(characters[current.item()])
assert len(output) == 13
assert set(output) <= set(characters)
print("Generated:", "".join(output))
```

Even perfect transition estimates cannot express long-range dependencies with this architecture. A model that only sees the previous character cannot distinguish two histories ending in the same character. Longer context requires a different representation and architecture, not simply more epochs.

## SOLO exercise — Make the shift a reusable contract

Implement `next_token_pairs(text, vocabulary)`, returning two `torch.long` tensors containing all adjacent input/target pairs. Reject non-string text with `TypeError`, fewer than two characters with `ValueError`, and let an unknown character raise `KeyError`. Do not add an artificial end token or silently discard unknown characters.

```python exercise
def next_token_pairs(text, vocabulary):
    raise NotImplementedError("Build shifted input and target tensors")

example_x, example_y = next_token_pairs("aba", {"a": 0, "b": 1})
assert example_x.tolist() == [0, 1]
assert example_y.tolist() == [1, 0]
```

**Optional experiment:** hold out a different string, measure its loss, then compare with training loss. Change the corpus while preserving the vocabulary. Explain why a character absent from training needs an explicit unknown-token or vocabulary policy.

**Completion gate:** produce shifted pairs, reduce the observed training loss, generate a bounded continuation, and explain why this is a character bigram experiment rather than a tiny GPT. Reference answer: `solutions/draft/ch21.py`; tests: `tests/draft/test_ch21.py`.

Primary references: [PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html), [cross-entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html), and [Hugging Face generation strategies](https://huggingface.co/docs/transformers/generation_strategies). Context windows and causal attention are developed in Study 20; the small objective here also applies when an architecture uses a longer context.

---

## Study 22 — Adapt fewer parameters with LoRA

New first-draft study. Source notebook: `book/draft/22-parameter-efficient-adaptation.ipynb`.

# Study 22 — Adapt fewer parameters with LoRA

**First draft; optional adaptation study.** Full fine-tuning updates an existing model's parameters. When the base model is large, storing gradients and optimizer state for every weight can be expensive. Low-Rank Adaptation, or **LoRA**, learns a structured update while keeping the original weights frozen. This study makes that update concrete with one small linear layer on CPU. It does not fine-tune a downloaded pretrained model.

## Represent an update with two matrices

A linear layer with weight `W` maps a row vector `x` to `x @ W.T`. Suppose `W` has shape `[output_features, input_features]`. LoRA represents an update as `B @ A`, where `A` has shape `[rank, input_features]` and `B` has shape `[output_features, rank]`. The **rank** controls the size of the intermediate representation and limits the kinds of update the adapter can express.

The adapted operation is `x @ W.T + (alpha / rank) * x @ A.T @ B.T`. `alpha` is a scaling setting; it is not the optimizer learning rate. This factorization uses `rank * (input_features + output_features)` adapter parameters. It saves parameters only when that count is smaller than the full weight size. Small layers and excessive rank can erase the saving.

```python
import torch
from torch import nn

torch.manual_seed(13)
torch.set_num_threads(1)
base = nn.Linear(12, 8, bias=False)
base.requires_grad_(False)
rank, alpha = 2, 4.0
a = nn.Parameter(torch.randn(rank, 12) * 0.01)
b = nn.Parameter(torch.zeros(8, rank))
x = torch.randn(6, 12)

def adapted(inputs):
    return base(inputs) + (alpha / rank) * (inputs @ a.T @ b.T)

torch.testing.assert_close(adapted(x), base(x))
assert not base.weight.requires_grad
assert a.numel() + b.numel() == 40 < base.weight.numel() == 96
print("Frozen base parameters:", base.weight.numel(), "Adapter parameters:", 40)
```

Zero-initializing `B` makes the initial update zero. `A` can start nonzero, allowing `B` to receive a gradient immediately. At that starting point, `A` may initially have zero gradient because the update also contains `B`; that is expected, not evidence that autograd is broken.

## Check the actual trainable boundary

We will ask the adapter to shift the base outputs toward an artificial target. Save a copy of the frozen weight before optimizing, then compare afterward. Checking `requires_grad` alone is weaker than checking that the promised base values remain unchanged.

```python
before = base.weight.detach().clone()
target = base(x).detach() + 0.25
optimizer = torch.optim.Adam([a, b], lr=0.05)
initial = torch.mean((adapted(x) - target) ** 2).item()
for _ in range(40):
    optimizer.zero_grad()
    loss = torch.mean((adapted(x) - target) ** 2)
    loss.backward()
    optimizer.step()
final = torch.mean((adapted(x) - target) ** 2).item()
torch.testing.assert_close(base.weight, before)
assert final < initial
print("Observed fitting loss:", initial, "->", final)
```

This verifies a low-rank optimization mechanism on six synthetic inputs. It says nothing about task quality on a language model or the best adapter rank. Real experiments must measure held-out behavior, memory, speed, and checkpoint compatibility on their actual models.

## SOLO exercise — Merge without changing the base

Implement `merged_weight(weight, a, b, *, alpha)`. Return a new tensor equal to the base plus the scaled adapter update. Require two-dimensional matrices, positive rank, and compatible dimensions; raise `ValueError` otherwise. Preserve the original tensors. Device and dtype compatibility are caller responsibilities, and PyTorch errors may propagate for incompatible devices or numeric types.

```python exercise
def merged_weight(weight, a, b, *, alpha):
    raise NotImplementedError("Merge the scaled low-rank update")

merged = merged_weight(torch.ones(3, 4), torch.ones(1, 4), torch.zeros(3, 1), alpha=1)
torch.testing.assert_close(merged, torch.ones(3, 4))
```

Compare a merged forward pass with the separate base-plus-adapter expression on the same input. Merging is useful for inference when the adapter should become part of the effective weight. Keep the original base and adapter metadata if you need to switch tasks or reproduce training. A merged tensor does not by itself retain an adapter's provenance.

**Optional extension:** configure LoRA through Hugging Face PEFT on the small pretrained fine-tuning workflow from Study 09. Check the installed PEFT version, choose actual target modules in that model, and inspect the trainable parameter names before training. That extension requires model downloads and is not executed by the offline example above.

**Completion gate:** explain the two matrix shapes, confirm the zero-update initialization, show that the base stays frozen during optimization, and verify merged versus separate forward equivalence. Reference: `solutions/draft/ch22.py`; tests: `tests/draft/test_ch22.py`.

Primary sources: the [original LoRA paper](https://arxiv.org/abs/2106.09685), [PEFT LoRA conceptual guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora), and [PEFT quicktour](https://huggingface.co/docs/peft/quicktour). The conceptual-guide link targets PEFT's main documentation; use the version matching your installed package for extension code.

---

## Study 23 — Study predictive error and replay

New first-draft study. Source notebook: `book/draft/23-research-architectures.ipynb`.

# Study 23 — Turn predictive error and replay into experiments

**First draft; optional research study.** Biologically inspired terminology is useful only when it leads to an explicit mechanism and a testable prediction. This study examines predictive error, replay, and a toy eligibility trace without claiming that these short programs reproduce the brain or replace established ML methods.

**Predictive coding** describes a family of models in which predictions and mismatches influence inference and learning. A prediction-error signal alone does not establish that an implementation is a predictive-coding network: the architecture, state updates, objective, and learning rule all matter. Our first experiment is simply online linear prediction, which gives us a controlled baseline for investigating error and replay.

## Begin with an observable failure

A scalar model predicts `weight * input`. First it sees examples of one relation, then examples of an opposing relation. Updating on recent data can worsen performance on earlier data. We can measure that loss rather than describe it vaguely as forgetting.

```python
def update(weight, sample, rate=0.1):
    value, target = sample
    error = target - weight * value
    return weight + rate * error * value

def mean_error(weight, samples):
    return sum((target - weight * value) ** 2 for value, target in samples) / len(samples)

old_task = [(1.0, 1.0)] * 12
new_task = [(1.0, -1.0)] * 12
weight = 0.0
for sample in old_task:
    weight = update(weight, sample)
old_error_before = mean_error(weight, old_task)
for sample in new_task:
    weight = update(weight, sample)
old_error_after = mean_error(weight, old_task)
assert old_error_after > old_error_before
print("Earlier-task error:", old_error_before, "->", old_error_after)
```

This example is intentionally conflicting: the same input has two targets and no task identity. One scalar weight cannot satisfy both. Replay cannot repair an impossible representation; it can only change the compromise. That distinction keeps an apparent improvement from being mistaken for a complete memory solution.

## Add replay as one controlled change

**Replay** revisits stored examples during learning. The buffer policy, sampling method, and additional update budget are part of the algorithm. Here we alternate new samples with one old sample and explicitly count the extra work. We keep the model and update rule fixed.

```python
replayed_weight = 0.0
for sample in old_task:
    replayed_weight = update(replayed_weight, sample)
for sample in new_task:
    replayed_weight = update(replayed_weight, sample)
    replayed_weight = update(replayed_weight, old_task[0])
print("No replay old/new:", mean_error(weight, old_task), mean_error(weight, new_task))
print("With replay old/new:", mean_error(replayed_weight, old_task), mean_error(replayed_weight, new_task))
assert mean_error(replayed_weight, old_task) < old_error_after
```

The replay run uses more updates. A fairer follow-up compares equal update budgets and reports both task errors. An **ablation** removes one mechanism while holding the important alternatives fixed. It helps determine whether the mechanism, extra compute, or another change explains a result. No single tiny experiment settles a biological hypothesis.

## A delayed learning signal

An **eligibility trace** keeps a fading record of recent local activity. A later **modulator** can gate or scale its effect on a weight. A toy discrete rule is `trace = decay * trace + activity`, followed by `weight += learning_rate * modulator * trace`. These variables are abstract numbers, not measured synapses or dopamine concentrations.

```python
trace, toy_weight = 0.0, 0.0
for activity, modulator in [(2.0, 0.0), (0.0, 1.0)]:
    trace = 0.5 * trace + activity
    toy_weight += 0.1 * modulator * trace
assert toy_weight == 0.1
print("Delayed gated update:", toy_weight)
```

The first event creates eligibility without changing the weight; the second event uses what remains. A longer delay reduces the contribution under this rule. This is a useful implementation exercise, not a detailed model of the experimental systems discussed in the research literature.

## SOLO exercise — Record the gated updates

Implement `eligibility_updates(events, *, decay=0.5, learning_rate=0.1)`. Accept an iterable of `(activity, modulator)` numeric pairs and return the weight after each event, starting with zero trace and zero weight. Reject decay outside `[0, 1]` or a negative learning rate with `ValueError`; other numeric validity is a caller obligation. Empty input returns an empty list.

```python exercise
def eligibility_updates(events, *, decay=0.5, learning_rate=0.1):
    raise NotImplementedError("Record the delayed gated weight history")

assert eligibility_updates([(2, 0), (0, 1)]) == [0.0, 0.1]
```

**Completion gate:** write a hypothesis, the observation that would refute it, and the controlled baseline. Implement the trace; compare zero and nonzero decay. Optional next experiment: add a task-ID feature and observe whether the earlier contradiction becomes representable. Reference answer: `solutions/draft/ch23.py`; tests: `tests/draft/test_ch23.py`.

Primary reading: [Whittington and Bogacz's predictive-coding network paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/), [Gerstner and colleagues on eligibility traces and three-factor learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/), and [PyTorch's replay-memory implementation in its DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). These sources describe different settings; this study borrows testable mechanisms without treating them as interchangeable theories.

---

## Study 24 — Keep provider integrations replaceable

New first-draft study. Source notebook: `book/draft/24-ai-platform-adapters.ipynb`.

# Study 24 — Keep AI provider integrations replaceable

**First draft; optional platform study.** An application may call a local model, OpenAI, or Claude, while developers use Copilot or another coding assistant to build it. These are different responsibilities. A coding assistant's presence in your editor should not become an undocumented dependency of your deployed application.

An **adapter** translates between an application contract and one provider's request/response format. The contract should describe the result your workflow needs, not expose every provider-specific response field. This study builds and tests that boundary entirely offline. It does not call an external model, require credentials, or demonstrate actual provider performance.

## Start from the application's decision

Our document workflow needs a category and a brief explanation. The application accepts exactly two fields, `label` and `reason`; labels come from a small allowlist. A **structured output** is a result that follows a defined schema. Even when a provider offers schema-constrained generation, the application should still check the result at its boundary.

```python
import json
from typing import Protocol

class RouteProvider(Protocol):
    def route(self, text: str) -> str:
        """Return JSON text with label and reason fields."""

class RecordedProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def route(self, text):
        self.calls += 1
        return self.response

provider = RecordedProvider('{"label":"invoice","reason":"Recorded fixture"}')
raw = provider.route("Invoice attached")
assert json.loads(raw)["label"] == "invoice"
assert provider.calls == 1
print("Offline adapter returned a recorded response.")
```

`Protocol` is a typing description of the methods a compatible object provides. It does not enforce a network policy or validate returned JSON. The fake deliberately ignores the input because its purpose is to test our application boundary using recorded cases. It is not an intelligent classifier.

## Retry only the failures you mean to retry

Temporary transport failures may justify retrying, but malformed data, authentication errors, and exhausted budgets should not trigger an unlimited loop. Retrying a request can also repeat a side effect. Keep model requests and tool execution separate, and apply idempotency rules to actions that change state.

```python
class FlakyProvider:
    def __init__(self):
        self.calls = 0

    def route(self, text):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("Synthetic transport timeout")
        return '{"label":"review","reason":"Needs a human"}'

def request_route(client, text, *, attempts=2):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return client.route(text)
        except TimeoutError:
            if attempt + 1 == attempts:
                raise

client = FlakyProvider()
assert json.loads(request_route(client, "A document"))["label"] == "review"
assert client.calls == 2
print("One explicit transient failure, then success.")
```

This example has no waiting because it is a deterministic test. A real adapter should respect provider retry guidance, timeouts, rate limits, and a total deadline. Validate whether an operation is safe to retry before adding backoff. Log request identifiers and error categories without exposing document contents or credentials.

## SOLO exercise — Validate the result, not the provider name

Write `parse_route(payload, allowed_labels=("claim", "invoice", "review"))`. Accept JSON text only; reject another Python type with `TypeError`. Require a dictionary containing exactly `label` and `reason`, require a label in the allowlist, and require a string reason. Schema failures raise `ValueError`; malformed JSON may propagate `json.JSONDecodeError`. Return the parsed dictionary.

```python exercise
def parse_route(payload, allowed_labels=("claim", "invoice", "review")):
    raise NotImplementedError("Validate the narrow route result schema")

assert parse_route('{"label":"claim","reason":"fixture"}')["label"] == "claim"
```

An explanation containing instructions remains data. Passing schema validation is not authorization to execute that text. Similarly, **tool calling** means a model can request a named operation with arguments; your application must select an allowlisted operation, validate arguments, apply authorization, execute it, and return a result. The model does not acquire unrestricted authority by naming a tool.

OpenAI and Claude have different wire formats for these exchanges. Translate them in separate adapters while sharing application-level tests. MCP can standardize a tool-service boundary, but it does not replace the application's access rules or result validation. Copilot instruction files guide development-time behavior; they are not runtime service credentials.

**Optional extension:** add one real provider adapter using that provider's current SDK and explicit credentials from the environment. Keep it disabled by default, record cost/usage separately, and run the same malformed-response tests against recorded fixtures. No live provider calls were executed for this draft.

**Completion gate:** demonstrate valid, malformed, extra-field, and invalid-label cases; show a bounded retry and explain why tool output is data rather than executable authority. Answer: `solutions/draft/ch24.py`; tests: `tests/draft/test_ch24.py`.

Primary references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [Claude tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview), [Python Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol), and [GitHub Copilot repository instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot).

---

## Study 25 — Learn orchestration by producing this book

New first-draft study. Source notebook: `book/draft/25-book-agent-pipeline.ipynb`.

# Study 25 — Learn orchestration by producing this book

**First draft; optional applied automation study.** Producing a book with coding agents is an engineering workflow: requirements become tasks, tasks produce artifacts, checks produce evidence, and an integration stage decides what can advance. It is also a useful way to study the orchestration skills used in ML pipelines.

The repository's reusable entry point is `tools/book_factory.py`; its operating guide is `docs/BOOK_FACTORY.md`. This study first demonstrates its underlying ideas with a small offline program. The example does not start agents, create threads, or call a hosted model. A real local run requires an installed and authenticated agent CLI and sufficient account usage allowance. Agents used inside this chat are a separate execution mechanism, not proof that the local CLI has been installed or tested.

## Represent dependencies as data

A **manifest** records what each task owns and requires. A **directed acyclic graph**, or DAG, describes prerequisites without circular dependencies. Some authoring can proceed in parallel once the shared outline and contracts exist. Validation and integration still depend on finished artifacts; making all stages simultaneous would only hide those dependencies.

```python
from graphlib import TopologicalSorter, CycleError

dependencies = {
    "outline": set(),
    "author-04": {"outline"},
    "author-05": {"outline"},
    "validate": {"author-04", "author-05"},
    "review": {"validate"},
}
order = list(TopologicalSorter(dependencies).static_order())
assert order.index("outline") < order.index("author-04")
assert order.index("author-05") < order.index("validate")
assert order[-1] == "review"
print("One valid stage order:", order)
```

A valid order is not necessarily unique. The two authors can run concurrently, subject to a worker limit. The important constraint is that both finish before validation starts. A cycle such as “A needs B, B needs A” should fail before spending agent time.

## Persist what was actually attempted

An in-memory `done` flag disappears when a process stops. Persist a small state record containing a schema version, selected task, requirements fingerprint, status, and evidence paths. Write to a temporary sibling and atomically replace the state file so readers do not observe a partially written JSON document.

```python
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as temporary:
    directory = Path(temporary)
    state_path = directory / "state.json"
    pending = directory / "state.pending.json"
    state = {"schema_version": 1, "chapter": "04", "status": "validated",
             "checks": {"tests": "passed", "editorial_review": "pending"}}
    pending.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(pending, state_path)
    restored = json.loads(state_path.read_text(encoding="utf-8"))
    assert restored["status"] == "validated"
    assert restored["checks"]["editorial_review"] == "pending"
print("State survived a fresh read; review is still pending.")
```

Atomic replacement helps preserve a complete state record; it does not make every artifact write transactional or guarantee durability after every possible machine failure. A resumed run should verify artifact presence and content fingerprints before trusting earlier success. “The process exited with zero” and “the chapter meets the learning objective” are different claims.

## SOLO exercise — Decide whether cached work still applies

Implement `task_fingerprint(chapter, shared_context)`. Both inputs must be dictionaries, otherwise raise `TypeError`. Serialize an object containing those two values with sorted dictionary keys, UTF-8, and compact separators, then return its SHA-256 hexadecimal digest. Preserve list order because ordered objectives may carry meaning. Values must be JSON-serializable; standard serialization errors may propagate.

```python exercise
def task_fingerprint(chapter, shared_context):
    raise NotImplementedError("Fingerprint task and shared requirements")

assert task_fingerprint({"id":"04", "topic":"tensor"}, {"style":"short"}) == task_fingerprint({"topic":"tensor", "id":"04"}, {"style":"short"})
assert task_fingerprint({"id":"04"}, {"style":"short"}) != task_fingerprint({"id":"04"}, {"style":"long"})
```

A hash detects changed inputs, not changed meaning with certainty. Include the actual guidance text or its content digest, not only its filename. If a dependency's exported interface changes, invalidate its dependants according to the dependency contract. Do not regenerate unrelated chapters merely because another chapter's punctuation changed.

## From the exercise to the actual factory

Use `python tools/book_factory.py plan --chapter 04` to inspect a chapter assignment before dispatch. The manifest defines exact manuscript, notebook, solution, and test paths. Plan output should make the selected task and requirements visible. Consult the guide for full-run, request-routing, validation, status, and resume options; these are runnable tooling instructions, not an invitation to bypass its checks.

Git worktrees give authors separate checkouts while sharing repository history. Ownership still matters: an isolated author could edit a shared guide, so the runner must compare changed paths with the task's allowlist. Preserve failed worktrees and logs for diagnosis. An author should not approve its own prose merely because tests pass. A separate review examines explanations, exercise difficulty, correctness, and scope before candidate integration. Promotion or merging remains an explicit integration decision.

**Completion gate:** inspect a plan, reproduce a dependency cycle failure, show a state record surviving a reload, and demonstrate fingerprint invalidation after a changed requirement. Reference: `solutions/draft/ch25.py`; tests: `tests/draft/test_ch25.py`. Optional extension: use a fake CLI command to force author failure and verify that the pipeline records failure without promoting its output.

Primary references: [Python graphlib](https://docs.python.org/3/library/graphlib.html), [Git worktrees](https://git-scm.com/docs/git-worktree), [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive), and [Python os.replace](https://docs.python.org/3/library/os.html#os.replace).

---

## Study 26 — Deliver an end-to-end ML/AI workflow

New first-draft study. Source notebook: `book/draft/26-integrated-capstone.ipynb`.

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
