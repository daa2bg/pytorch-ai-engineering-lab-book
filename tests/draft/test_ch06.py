import math

import pytest
import torch
from torch import nn

from solutions.draft import ch06
from solutions.draft.ch06 import DigitMLP, evaluate, train_candidate, training_step


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_training_step_changes_parameters_and_reduces_fixture_loss():
    torch.manual_seed(7)
    model = nn.Linear(2, 2)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    x = torch.tensor([[-1., -1.], [1., 1.]])
    y = torch.tensor([0, 1])
    originals = x.clone(), y.clone()
    before = model.weight.detach().clone()
    with torch.no_grad():
        pre_update_loss = nn.functional.cross_entropy(model(x), y).item()
    first = training_step(model, optimizer, x, y)
    assert model.training
    assert isinstance(first, float)
    assert first == pytest.approx(pre_update_loss)
    assert not torch.equal(model.weight, before)
    assert torch.equal(x, originals[0]) and torch.equal(y, originals[1])
    for _ in range(30):
        last = training_step(model, optimizer, x, y)
    assert last < first
    with torch.no_grad():
        assert torch.equal(model(x).argmax(dim=1), y)


def test_training_step_clears_stale_gradients():
    torch.manual_seed(5)
    model = nn.Linear(1, 2)
    twin = nn.Linear(1, 2)
    twin.load_state_dict(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 100.)
    x, y = torch.tensor([[-1.], [1.]]), torch.tensor([0, 1])
    training_step(model, torch.optim.SGD(model.parameters(), lr=0.1), x, y)
    training_step(twin, torch.optim.SGD(twin.parameters(), lr=0.1), x, y)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), twin.parameters()))


def test_backward_without_step_has_gradients_but_no_parameter_change():
    torch.manual_seed(7)
    model = nn.Linear(2, 2)
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.tensor([[-1., -1.], [1., 1.]])
    y = torch.tensor([0, 1])
    nn.functional.cross_entropy(model(x), y).backward()
    assert any(p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    training_step(model, torch.optim.SGD(model.parameters(), lr=0.2), x, y)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_evaluation_does_not_update_parameters_or_gradients():
    torch.manual_seed(3)
    model = DigitMLP()
    features = torch.randn(5, 64)
    targets = torch.tensor([0, 1, 2, 3, 4])
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    weights = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad.clone() for p in model.parameters()]
    result = evaluate(model, features, targets)
    assert not model.training
    assert math.isfinite(result["loss"])
    assert 0 <= result["accuracy"] <= 1
    assert all(torch.equal(a, b) for a, b in zip(weights, model.parameters()))
    assert all(torch.equal(a, p.grad) for a, p in zip(gradients, model.parameters()))


def test_candidate_uses_every_training_row_each_epoch_and_weights_tail(monkeypatch):
    # Instrument only the update boundary: row IDs remain paired under shuffling,
    # the same loader traverses anew, and validation rows never reach an update.
    features = torch.zeros(5, 64)
    features[:, 0] = torch.arange(5)
    labels = torch.arange(5)
    seen = []
    sizes = []

    def observe_step(model, optimizer, batch_x, batch_y):
        assert torch.equal(batch_x[:, 0].long(), batch_y)
        seen.extend(batch_y.tolist())
        sizes.append(len(batch_y))
        return batch_y.float().mean().item()

    monkeypatch.setattr(ch06, "training_step", observe_step)
    _, records = train_candidate(
        features, labels, torch.zeros(2, 64), torch.tensor([8, 9]),
        epochs=2, batch_size=2,
    )
    assert sizes == [2, 2, 1, 2, 2, 1]
    assert sorted(seen[:5]) == sorted(seen[5:]) == list(range(5))
    assert [r["train_loss"] for r in records] == pytest.approx([2.0, 2.0])


def test_candidate_is_reproducible_and_validation_cannot_change_training():
    torch.manual_seed(8)
    train_x, val_x = torch.rand(13, 64), torch.rand(4, 64)
    train_y, val_y = torch.arange(13) % 10, torch.arange(4)
    originals = [x.clone() for x in (train_x, train_y, val_x, val_y)]
    first, first_records = train_candidate(train_x, train_y, val_x, val_y, epochs=2)
    twin, twin_records = train_candidate(train_x, train_y, val_x, val_y, epochs=2)
    changed_val, changed_records = train_candidate(
        train_x, train_y, val_x * 2, (val_y + 1) % 10, epochs=2,
    )
    assert first_records == twin_records
    assert first_records != changed_records
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), twin.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), changed_val.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(originals, (train_x, train_y, val_x, val_y)))
    assert first(val_x).shape == (4, 10)
    assert any(isinstance(layer, nn.ReLU) for layer in first.modules())
    assert sum(p.numel() for p in first.parameters()) == 1210


def test_candidate_really_learns_bundled_digits():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    train_ids, held_ids = train_test_split(
        list(range(len(digits.target))), test_size=0.4,
        random_state=7, stratify=digits.target,
    )
    val_ids, test_ids = train_test_split(
        held_ids, test_size=0.5, random_state=7,
        stratify=digits.target[held_ids],
    )
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    train_x = torch.tensor(digits.data[train_ids] / 16, dtype=torch.float32)
    train_y = torch.tensor(digits.target[train_ids], dtype=torch.long)
    val_x = torch.tensor(digits.data[val_ids] / 16, dtype=torch.float32)
    val_y = torch.tensor(digits.target[val_ids], dtype=torch.long)
    torch.manual_seed(7)
    untrained = DigitMLP()
    initial_validation = evaluate(untrained, val_x, val_y)
    model, records = train_candidate(train_x, train_y, val_x, val_y)
    assert records[-1]["train_loss"] < records[0]["train_loss"]
    assert records[-1]["val_loss"] < initial_validation["loss"]
    assert all(math.isfinite(row["train_loss"]) for row in records)
    assert any(not torch.equal(a, b) for a, b in zip(untrained.parameters(), model.parameters()))
    # No architecture ranking or brittle accuracy threshold is a pass condition.
