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
