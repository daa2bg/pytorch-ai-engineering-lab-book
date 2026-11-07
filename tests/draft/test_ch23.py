import pytest
from solutions.draft.ch23 import eligibility_updates


def test_delayed_modulator_uses_decayed_trace():
    assert eligibility_updates([(2, 0), (0, 1)], decay=0.5) == pytest.approx([0, 0.1])
    assert eligibility_updates([(2, 0), (0, 1)], decay=0) == [0, 0]


def test_zero_modulation_and_empty_events():
    assert eligibility_updates([(1, 0), (2, 0)]) == [0, 0]
    assert eligibility_updates([]) == []


@pytest.mark.parametrize("decay,rate", [(-1, 1), (2, 1), (0.5, -1)])
def test_reject_invalid_hyperparameters(decay, rate):
    with pytest.raises(ValueError):
        eligibility_updates([], decay=decay, learning_rate=rate)
