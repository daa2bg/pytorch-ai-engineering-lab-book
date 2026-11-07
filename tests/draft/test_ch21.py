import pytest
torch = pytest.importorskip("torch")
from solutions.draft.ch21 import next_token_pairs


def test_pairs_are_shifted_and_integer():
    x, y = next_token_pairs("aba", {"a": 0, "b": 1})
    assert x.tolist() == [0, 1]
    assert y.tolist() == [1, 0]
    assert x.dtype == y.dtype == torch.long


@pytest.mark.parametrize("text", ["", "a"])
def test_pairs_need_an_observation(text):
    with pytest.raises(ValueError):
        next_token_pairs(text, {"a": 0})


def test_pairs_reject_bad_type_and_unknown_character():
    with pytest.raises(TypeError):
        next_token_pairs(17, {})
    with pytest.raises(KeyError):
        next_token_pairs("ab", {"a": 0})
