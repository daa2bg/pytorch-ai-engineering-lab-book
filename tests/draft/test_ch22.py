import pytest
torch = pytest.importorskip("torch")
from solutions.draft.ch22 import merged_weight


def test_merge_matches_separate_forward_and_preserves_base():
    torch.manual_seed(2)
    w, a, b, x = torch.randn(3, 4), torch.randn(2, 4), torch.randn(3, 2), torch.randn(5, 4)
    before = w.clone()
    merged = merged_weight(w, a, b, alpha=4)
    torch.testing.assert_close(x @ merged.T, x @ w.T + 2 * (x @ a.T @ b.T))
    torch.testing.assert_close(w, before)
    assert merged.data_ptr() != w.data_ptr()


def test_zero_b_preserves_function():
    w = torch.ones(3, 4)
    torch.testing.assert_close(merged_weight(w, torch.ones(1, 4), torch.zeros(3, 1), alpha=1), w)


@pytest.mark.parametrize("a,b", [(torch.ones(0, 4), torch.ones(3, 0)), (torch.ones(2, 5), torch.ones(3, 2))])
def test_merge_rejects_bad_shapes(a, b):
    with pytest.raises(ValueError):
        merged_weight(torch.ones(3, 4), a, b, alpha=1)
