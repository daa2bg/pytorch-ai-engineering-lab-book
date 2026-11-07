"""Study 06 answers: repair an update and train a smaller digits MLP.

Compare only after completing the SOLO exercises. Validation may guide a width
choice; this API deliberately has no test-set argument. It expects nonempty,
aligned CPU tensors and positive integer width/epoch/batch settings as documented
in the chapter. PyTorch reports malformed tensor contracts.
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def training_step(model, optimizer, features, targets):
    """Return pre-update cross-entropy while changing parameters exactly once.

    backward() populates gradients but cannot update weights. Clear stale gradients
    before the forward pass and call step() afterward. Input tensors stay untouched.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(features)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.item()


class DigitMLP(nn.Module):
    """Hidden features require a nonlinearity; two affine layers alone collapse."""

    def __init__(self, hidden=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10)
        )

    def forward(self, features):
        return self.layers(features)


def evaluate(model, features, targets):
    """Evaluate a fixed model; mode selection and gradient tracking are separate."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = nn.functional.cross_entropy(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}


def train_candidate(train_x, train_y, val_x, val_y, *,
                    hidden=16, epochs=30, batch_size=64, seed=7):
    """Train a new MLP and return (model, per-epoch records).

    Reset initialization and loader RNG separately, so layer width does not silently
    change the shuffle order. The loader itself remains reusable across epochs.
    Weight each mean batch loss by its sample count, preserving the short tail.
    The returned model is in evaluation mode after the final validation pass.

    Compare this 16-unit candidate with the worked 32-unit and linear models at
    the same budget. Reduced width means fewer parameters, not a guaranteed loss
    of accuracy: initialization, optimization and sampling also affect the result.
    A validation tie supports the simpler candidate, subject to error inspection.
    """
    torch.manual_seed(seed)
    model = DigitMLP(hidden)
    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0, drop_last=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    records = []
    for epoch in range(epochs):
        loss_total, seen = 0.0, 0
        for features, targets in loader:
            loss = training_step(model, optimizer, features, targets)
            loss_total += loss * len(targets)
            seen += len(targets)
        validation = evaluate(model, val_x, val_y)
        records.append({
            "epoch": epoch + 1, "train_loss": loss_total / seen,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        })
    return model, records
