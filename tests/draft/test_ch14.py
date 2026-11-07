import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft14", Path(__file__).resolve().parents[2] / "solutions/draft/ch14.py")
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


def test_recall_only_counts_relevant_unique_top_k_ids():
    assert answer.recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 0.5
    assert answer.recall_at_k(["wrong", "b", "a"], ["a", "b"], 3) == 1.0
    assert answer.recall_at_k([], ["a"], 5) == 0.0


@pytest.mark.parametrize("relevant,k", [([], 1), (["a"], 0), (["a"], -1)])
def test_undefined_recall_contract_is_rejected(relevant, k):
    with pytest.raises(ValueError):
        answer.recall_at_k(["a"], relevant, k)
