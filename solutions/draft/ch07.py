"""Chapter 07 answers: reproducible epoch-boundary resume plus legacy counts.

The CPU contract uses trusted, locally created checkpoints; it does not resume
mid-batch, validate arbitrary model files, or guarantee cross-version bitwise runs.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_model(config):
    if config["kind"] == "linear":
        return nn.Linear(config["input_features"], config["classes"])
    if config["kind"] != "mlp":
        raise ValueError("unsupported model kind")
    return nn.Sequential(
        nn.Linear(config["input_features"], config["hidden"]), nn.ReLU(),
        nn.Linear(config["hidden"], config["classes"]),
    )


def make_run(config):
    torch.manual_seed(config["seed"])
    model = build_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"])
    generator = torch.Generator().manual_seed(config["seed"])
    return model, optimizer, generator


def train_epochs(model, optimizer, generator, features, targets, config,
                 start_epoch, total_epochs):
    """Return post-start epoch mean training losses, including every tail sample."""
    loader = DataLoader(TensorDataset(features, targets),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    losses = []
    for _ in range(start_epoch, total_epochs):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_y)
            seen += len(batch_y)
        losses.append(total / seen)
    return losses


def save_checkpoint(path, model, optimizer, generator, epoch, losses, config,
                    data_contract):
    """Snapshot all state after a complete epoch; leave no live tensor aliases."""
    checkpoint = {
        "schema_version": 1, "epoch": epoch, "losses": list(losses),
        "config": deepcopy(config), "data_contract": deepcopy(data_contract),
        "model_state": deepcopy(model.state_dict()),
        "optimizer_state": deepcopy(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().clone(),
        "loader_rng": generator.get_state().clone(),
    }
    torch.save(checkpoint, Path(path))


def restore_run(path):
    """Restore the next update, not only the next inference result.

    Construct objects first: initialization itself consumes torch RNG. Then load
    parameters and momentum buffers, and finally restore both RNG positions.
    A seed alone would restart the shuffle stream at epoch zero.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    config = checkpoint["config"]
    model, optimizer, generator = make_run(config)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    generator.set_state(checkpoint["loader_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    return model, optimizer, generator, checkpoint


def resume_training(path, train_x, train_y, total_epochs):
    """Reference SOLO repair: continue exactly to the total epoch budget.

    Caller must supply the original ordered, scaled training rows described by
    data_contract; total_epochs is an integer >= completed checkpoint epochs.
    This function does not consume validation/test labels. It returns the live
    model, optimizer, loader generator and complete loss history. Inputs stay
    unchanged; training state changes. At the same CPU/version settings the
    uninterrupted and resumed runs should match exactly at an epoch boundary.
    """
    model, optimizer, generator, checkpoint = restore_run(path)
    if total_epochs < checkpoint["epoch"]:
        raise ValueError("total_epochs precedes saved epoch")
    losses = checkpoint["losses"] + train_epochs(
        model, optimizer, generator, train_x, train_y, checkpoint["config"],
        checkpoint["epoch"], total_epochs,
    )
    return model, optimizer, generator, losses


def binary_counts(predictions, targets):
    """Optional binary-metric exercise retained for earlier consumers."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal lengths")
    if any(value not in (0, 1) for value in predictions + targets):
        raise ValueError("expected only binary values 0 and 1")
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for predicted, actual in zip(predictions, targets):
        if predicted == 1:
            counts["tp" if actual == 1 else "fp"] += 1
        else:
            counts["fn" if actual == 1 else "tn"] += 1
    return counts
