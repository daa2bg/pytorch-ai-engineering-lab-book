"""Study 04 reference answer: per-feature centering with an explicit contract."""
import torch


def center_columns(x):
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("expected a nonempty [samples, features] tensor")
    if not x.is_floating_point():
        raise ValueError("expected floating-point storage")
    return x - x.mean(dim=0, keepdim=True)


def fit_standardizer(train_x):
    """Fit population statistics on training data only; constant columns use 1.

    Reusing these values prevents a sample's representation from depending on
    its prediction companions. The returned tensors own their statistics.
    """
    if train_x.ndim != 2 or 0 in train_x.shape or not train_x.is_floating_point():
        raise ValueError("expected a nonempty floating-point training table")
    mean = train_x.mean(dim=0, keepdim=True)
    spread = train_x.std(dim=0, correction=0, keepdim=True)
    scale = torch.where(spread == 0, torch.ones_like(spread), spread)
    return {"mean": mean, "scale": scale}


def apply_standardizer(x, state):
    """Apply a valid fitted state without changing it or learning from x."""
    mean, scale = state["mean"], state["scale"]
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("expected a floating-point [samples, features] table")
    if x.shape[1] != mean.shape[1] or x.dtype != mean.dtype or x.device != mean.device:
        raise ValueError("features, dtype and device must match fitted state")
    return (x - mean) / scale
