import importlib.util
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("draft19", Path(__file__).parents[2] / "solutions/draft/ch19.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_padding_cannot_change_pooled_value():
    values = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
    before = values.clone()
    assert torch.equal(m.masked_mean(values, torch.tensor([2])), torch.tensor([[2., 4.]]))
    assert torch.equal(values, before)

def test_bad_lengths_and_shape():
    with pytest.raises(ValueError): m.masked_mean(torch.ones(1, 2, 3), torch.tensor([0]))
    with pytest.raises(ValueError): m.masked_mean(torch.ones(2, 3), torch.tensor([1]))

def test_gradient_excludes_padding():
    values = torch.ones(1, 3, 2, requires_grad=True)
    m.masked_mean(values, torch.tensor([2])).sum().backward()
    assert torch.equal(values.grad[0, 2], torch.zeros(2))
