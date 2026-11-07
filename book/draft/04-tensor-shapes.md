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
