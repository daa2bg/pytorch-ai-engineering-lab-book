import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft15", Path(__file__).parents[2] / "solutions/draft/ch15.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_batches_preserve_order_and_tail():
    calls = []
    def predict(values):
        calls.append(list(values))
        return [v * 2 for v in values]
    assert m.batched_predict(iter(range(5)), predict, 2) == [0, 2, 4, 6, 8]
    assert calls == [[0, 1], [2, 3], [4]]

def test_empty_does_not_call_model():
    assert m.batched_predict([], lambda _: pytest.fail("unexpected call"), 2) == []

@pytest.mark.parametrize("size", [0, -1])
def test_bad_size(size):
    with pytest.raises(ValueError): m.batched_predict([1], lambda x: x, size)

def test_bad_result_count():
    with pytest.raises(ValueError): m.batched_predict([1, 2], lambda _: [1], 2)

@pytest.mark.parametrize("size", [True, 1.5, "2"])
def test_noninteger_size(size):
    with pytest.raises(TypeError): m.batched_predict([1], lambda x: x, size)
