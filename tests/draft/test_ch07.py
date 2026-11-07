import pytest

from solutions.draft.ch07 import binary_counts


def test_binary_counts_orientation_and_no_mutation():
    predictions, targets = [1, 1, 0, 0], [1, 0, 1, 0]
    assert binary_counts(predictions, targets) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert predictions == [1, 1, 0, 0] and targets == [1, 0, 1, 0]


def test_binary_counts_asymmetric_fixture():
    assert binary_counts([1, 1, 0], [0, 0, 1]) == {"tp": 0, "fp": 2, "fn": 1, "tn": 0}
    assert binary_counts([], []) == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


@pytest.mark.parametrize("predictions,targets", [([1], []), ([2], [1]), ([1], [-1]), ([None], [0])])
def test_binary_counts_rejects_bad_contract(predictions, targets):
    with pytest.raises(ValueError):
        binary_counts(predictions, targets)


@pytest.fixture
def digits_run():
    import torch
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    digits = load_digits()
    train, held = train_test_split(list(range(len(digits.target))), test_size=0.4,
                                  random_state=7, stratify=digits.target)
    val, test = train_test_split(held, test_size=0.5, random_state=7,
                                stratify=digits.target[held])
    assert (len(train), len(val), len(test)) == (1078, 359, 360)
    x = torch.tensor(digits.data[train] / 16, dtype=torch.float32)
    y = torch.tensor(digits.target[train], dtype=torch.long)
    config = {"kind": "mlp", "input_features": 64, "classes": 10, "hidden": 32,
              "learning_rate": 0.05, "momentum": 0.9, "batch_size": 64, "seed": 7}
    yield x, y, config
    torch.set_num_threads(previous)


def test_real_digits_resume_matches_uninterrupted_weights_losses_and_rng(tmp_path, digits_run):
    import torch
    from solutions.draft.ch07 import make_run, train_epochs, save_checkpoint, resume_training

    x, y, config = digits_run
    originals = x.clone(), y.clone()
    full, full_opt, full_gen = make_run(config)
    full_losses = train_epochs(full, full_opt, full_gen, x, y, config, 0, 6)
    full_rng, full_loader_rng = torch.get_rng_state(), full_gen.get_state()
    interrupted, opt, gen = make_run(config)
    losses = train_epochs(interrupted, opt, gen, x, y, config, 0, 3)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, interrupted, opt, gen, 3, losses, config, {"task_id":"digits.v1"})
    assert opt.state_dict()["state"]  # Actual momentum buffers, not an empty dict.
    torch.rand(17)  # Simulate unrelated work between interruption and restoration.
    resumed, resumed_opt, resumed_gen, resumed_losses = resume_training(path, x, y, 6)
    assert resumed_losses == full_losses
    assert all(torch.equal(a, b) for a, b in zip(full.parameters(), resumed.parameters()))
    assert torch.equal(torch.get_rng_state(), full_rng)
    assert torch.equal(resumed_gen.get_state(), full_loader_rng)
    for original, restored in zip(full_opt.state.values(), resumed_opt.state.values()):
        assert torch.equal(original["momentum_buffer"], restored["momentum_buffer"])
    assert torch.equal(x, originals[0]) and torch.equal(y, originals[1])
    assert full_losses[-1] < full_losses[0]


@pytest.mark.parametrize("omit", ["optimizer", "loader"])
def test_omitting_training_state_changes_next_epoch(tmp_path, digits_run, omit):
    import torch
    from solutions.draft.ch07 import make_run, train_epochs, save_checkpoint, restore_run

    x, y, config = digits_run
    model, opt, gen = make_run(config)
    losses = train_epochs(model, opt, gen, x, y, config, 0, 3)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, opt, gen, 3, losses, config, {})
    correct, correct_opt, correct_gen, _ = restore_run(path)
    train_epochs(correct, correct_opt, correct_gen, x, y, config, 3, 4)
    broken, broken_opt, broken_gen, _ = restore_run(path)
    if omit == "optimizer":
        broken_opt = torch.optim.SGD(broken.parameters(), lr=0.05, momentum=0.9)
    else:
        broken_gen.manual_seed(config["seed"])
    train_epochs(broken, broken_opt, broken_gen, x, y, config, 3, 4)
    assert any(not torch.equal(a, b) for a, b in zip(correct.parameters(), broken.parameters()))


def test_checkpoint_is_snapshot_and_restores_rng_after_construction(tmp_path, digits_run):
    import torch
    from solutions.draft.ch07 import make_run, train_epochs, save_checkpoint, restore_run

    x, y, config = digits_run
    model, opt, gen = make_run(config)
    losses = train_epochs(model, opt, gen, x, y, config, 0, 1)
    path = tmp_path / "snapshot.pt"
    saved_weights = [p.detach().clone() for p in model.parameters()]
    save_checkpoint(path, model, opt, gen, 1, losses, config, {"scaler":{"divisor":16.0}})
    expected_random = torch.rand(5)
    train_epochs(model, opt, gen, x, y, config, 1, 2)
    restored, _, _, checkpoint = restore_run(path)
    assert checkpoint["epoch"] == 1
    assert checkpoint["data_contract"]["scaler"]["divisor"] == 16.0
    assert torch.equal(torch.rand(5), expected_random)
    assert all(torch.equal(a, b) for a, b in zip(saved_weights, restored.parameters()))


def test_resume_rejects_earlier_budget(tmp_path, digits_run):
    from solutions.draft.ch07 import make_run, train_epochs, save_checkpoint, resume_training

    x, y, config = digits_run
    model, opt, gen = make_run(config)
    losses = train_epochs(model, opt, gen, x, y, config, 0, 1)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, opt, gen, 1, losses, config, {})
    with pytest.raises(ValueError, match="precedes"):
        resume_training(path, x, y, 0)
