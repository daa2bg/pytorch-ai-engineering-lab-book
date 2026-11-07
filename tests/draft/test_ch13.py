import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft13", Path(__file__).resolve().parents[2] / "solutions/draft/ch13.py")
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


@pytest.mark.parametrize("left,right,expected", [([1, 0], [2, 0], 1), ([1, 0], [0, 1], 0), ([1, 0], [-1, 0], -1), ([0, 0], [1, 2], 0)])
def test_cosine_values(left, right, expected):
    assert answer.cosine(left, right) == pytest.approx(expected)


@pytest.mark.parametrize("left,right", [([], []), ([1], [1, 2]), ([float("nan")], [1])])
def test_bad_vectors_rejected(left, right):
    with pytest.raises(ValueError):
        answer.cosine(left, right)


@pytest.mark.parametrize("scale", [1e308, 1e-300])
def test_finite_extreme_scales_preserve_direction(scale):
    left = [scale, 0.0]
    right = [scale, 0.0]
    assert answer.cosine(left, right) == pytest.approx(1.0)
    assert answer.cosine(left, [-scale, 0.0]) == pytest.approx(-1.0)
    assert left == right == [scale, 0.0]
