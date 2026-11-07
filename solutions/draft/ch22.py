"""Reference answer for merging a low-rank adapter into a linear weight."""
import torch


def merged_weight(weight, a, b, *, alpha):
    if weight.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("all weights must be matrices")
    rank = a.shape[0]
    if rank < 1 or b.shape[1] != rank or (b.shape[0], a.shape[1]) != tuple(weight.shape):
        raise ValueError("incompatible adapter shapes")
    return weight + (alpha / rank) * (b @ a)
