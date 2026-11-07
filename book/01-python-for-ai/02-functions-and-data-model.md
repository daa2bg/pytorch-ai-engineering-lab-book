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
