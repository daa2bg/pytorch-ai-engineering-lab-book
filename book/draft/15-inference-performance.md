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
