import pytest
import torch

from solutions.draft.ch04 import center_columns


def test_center_columns_values_ownership_and_dtype():
    x = torch.tensor([[1., 8.], [3., 8.]], dtype=torch.float64)
    before = x.clone()
    result = center_columns(x)
    assert torch.equal(result, torch.tensor([[-1., 0.], [1., 0.]], dtype=torch.float64))
    assert torch.equal(x, before)
    assert result.dtype == x.dtype and result.device == x.device
    assert result.data_ptr() != x.data_ptr()


def test_single_sample_centers_to_zero():
    assert torch.equal(center_columns(torch.tensor([[2., 3.]])), torch.zeros(1, 2))


@pytest.mark.parametrize("x", [torch.tensor([1.]), torch.empty(0, 2), torch.empty(2, 0), torch.ones(2, 2, dtype=torch.int64)])
def test_center_columns_rejects_invalid_contract(x):
    with pytest.raises(ValueError):
        center_columns(x)


def test_fit_apply_training_stats_and_batch_independence():
    from solutions.draft.ch04 import fit_standardizer, apply_standardizer

    training = torch.tensor([[1., 8.], [3., 8.]], dtype=torch.float64)
    future = torch.tensor([[5., 9.], [7., 10.]], dtype=torch.float64)
    originals = training.clone(), future.clone()
    state = fit_standardizer(training)
    assert torch.equal(state["mean"], torch.tensor([[2., 8.]], dtype=torch.float64))
    assert torch.equal(state["scale"], torch.ones(1, 2, dtype=torch.float64))
    result = apply_standardizer(future, state)
    assert torch.equal(result, torch.tensor([[3., 1.], [5., 2.]], dtype=torch.float64))
    assert torch.equal(apply_standardizer(future[:1], state), result[:1])
    assert torch.equal(training, originals[0]) and torch.equal(future, originals[1])
    assert result.dtype == future.dtype
    training.fill_(999.)
    assert state["mean"].tolist() == [[2., 8.]]


def test_singleton_fit_and_empty_prediction():
    from solutions.draft.ch04 import fit_standardizer, apply_standardizer

    state = fit_standardizer(torch.tensor([[2., 3.]]))
    assert state["scale"].tolist() == [[1., 1.]]
    assert apply_standardizer(torch.empty(0, 2), state).shape == (0, 2)


@pytest.mark.parametrize("bad", [torch.ones(2), torch.empty(0, 2), torch.empty(2, 0), torch.ones(2, 2, dtype=torch.long)])
def test_fit_rejects_bad_contract(bad):
    from solutions.draft.ch04 import fit_standardizer

    with pytest.raises(ValueError):
        fit_standardizer(bad)


@pytest.mark.parametrize("bad", [torch.ones(2), torch.ones(2, 3), torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.float64)])
def test_apply_rejects_incompatible_input(bad):
    from solutions.draft.ch04 import fit_standardizer, apply_standardizer

    state = fit_standardizer(torch.ones(2, 2))
    with pytest.raises(ValueError):
        apply_standardizer(bad, state)


def test_serialized_statistics_reproduce_outputs_and_refitting_does_not():
    import json
    from solutions.draft.ch04 import fit_standardizer, apply_standardizer

    state = fit_standardizer(torch.tensor([[1., 8.], [3., 8.]]))
    future = torch.tensor([[5., 9.], [7., 10.]])
    text = json.dumps({key: value.tolist() for key, value in state.items()})
    restored = {key: torch.tensor(value) for key, value in json.loads(text).items()}
    expected = apply_standardizer(future, state)
    assert torch.equal(apply_standardizer(future, restored), expected)
    assert not torch.equal(apply_standardizer(future, fit_standardizer(future)), expected)
