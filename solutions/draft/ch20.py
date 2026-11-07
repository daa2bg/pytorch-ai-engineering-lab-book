"""Reference answer: one causal scaled dot-product attention head."""
import math
import torch


def causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("expected rank-2 [T,D] tensors")
    if query.shape != key.shape or value.shape[0] != query.shape[0]:
        raise ValueError("query/key shapes and token counts must agree")
    if query.shape[0] == 0 or query.shape[1] == 0:
        raise ValueError("attention requires tokens and features")
    if not all(item.is_floating_point() for item in (query, key, value)):
        raise TypeError("attention inputs must be floating point")
    scores = query @ key.T / math.sqrt(query.shape[-1])
    allowed = torch.ones(scores.shape, dtype=torch.bool, device=scores.device).tril()
    weights = torch.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1)
    return weights @ value, weights
