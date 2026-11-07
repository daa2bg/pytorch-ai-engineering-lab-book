"""Reference answer: ignore padded sequence positions when pooling."""
import torch


def masked_mean(batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 3 or lengths.ndim != 1 or lengths.shape[0] != batch.shape[0]:
        raise ValueError("expected batch [B,T,D] and lengths [B]")
    if not batch.is_floating_point():
        raise TypeError("batch must be floating point")
    if lengths.dtype not in {torch.int32, torch.int64}:
        raise TypeError("lengths must be integer")
    if bool(((lengths <= 0) | (lengths > batch.shape[1])).any()):
        raise ValueError("lengths must lie in 1..T")
    lengths = lengths.to(batch.device)
    mask = torch.arange(batch.shape[1], device=batch.device)[None, :] < lengths[:, None]
    safe_values = batch.masked_fill(~mask[:, :, None], 0.0)
    return safe_values.sum(dim=1) / lengths[:, None]
