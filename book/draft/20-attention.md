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
