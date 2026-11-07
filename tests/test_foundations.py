import pytest
from ai_lab.foundations import batches, standardize

def test_batches_tail_and_order():
    assert list(batches(iter(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]

def test_empty_batches():
    assert list(batches([], 2)) == []

@pytest.mark.parametrize("size", [0, -1])
def test_bad_batch_size(size):
    with pytest.raises(ValueError):
        list(batches([1], size))

def test_standardize():
    out = standardize([1., 2., 3.])
    assert sum(out) == pytest.approx(0)
    assert sum(x*x for x in out)/3 == pytest.approx(1)

def test_constant_and_empty():
    assert standardize([4., 4.]) == [0., 0.]
    with pytest.raises(ValueError):
        standardize([])
