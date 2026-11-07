import importlib.util
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("draft20", Path(__file__).parents[2] / "solutions/draft/ch20.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_causal_mask_and_normalization():
    q = torch.eye(3)
    output, weights = m.causal_attention(q, q, torch.arange(6.).reshape(3, 2))
    assert output.shape == (3, 2)
    assert torch.equal(weights.triu(1), torch.zeros(3, 3))
    assert torch.allclose(weights.sum(-1), torch.ones(3))

def test_future_values_do_not_affect_prefix():
    q = torch.eye(3); values = torch.arange(6.).reshape(3, 2)
    before, _ = m.causal_attention(q, q, values)
    changed = values.clone(); changed[2] = 1000
    after, _ = m.causal_attention(q, q, changed)
    assert torch.allclose(before[:2], after[:2])
    assert not torch.allclose(before[2], after[2])

def test_invalid_shapes():
    with pytest.raises(ValueError): m.causal_attention(torch.ones(2, 3), torch.ones(3, 3), torch.ones(2, 3))
