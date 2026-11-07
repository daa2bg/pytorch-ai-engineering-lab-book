import pytest

from solutions.draft.ch05 import gradient_step


def test_gradient_step_matches_derivative_and_reduces_loss():
    updated = gradient_step(3., 1., 0.1)
    assert updated == pytest.approx(2.6)
    assert (updated - 1.) ** 2 < 4


def test_gradient_step_at_target_stays_at_target():
    assert gradient_step(-2., -2., 0.5) == -2.


@pytest.mark.parametrize("rate", [0., -0.1])
def test_gradient_step_rejects_nonpositive_rate(rate):
    with pytest.raises(ValueError):
        gradient_step(3., 1., rate)


def test_fit_gain_two_steps_excludes_stale_gradient_update():
    from solutions.draft.ch05 import fit_gain

    final, losses = fit_gain(2., 3., 5., 0.1, 2)
    assert final == pytest.approx(1.88)
    assert losses == pytest.approx([1., 0.64, 0.4096])
    assert final != pytest.approx(1.28)  # Accumulating stale gradients gives this.


@pytest.mark.parametrize("rate", [0.05, 0.1, 0.2])
def test_gain_trace_matches_analytic_error_recurrence(rate):
    from solutions.draft.ch05 import fit_gain

    final, losses = fit_gain(2., 3., 5., rate, 8)
    multiplier = 1 - 18 * rate
    assert len(losses) == 9
    assert losses == pytest.approx([multiplier ** (2 * step) for step in range(9)], rel=1e-12, abs=1e-12)
    assert final == pytest.approx((5 + multiplier ** 8) / 3)
    assert (losses[-1] < losses[0]) == (rate < 1 / 9)


def test_zero_input_cannot_change_parameter():
    from solutions.draft.ch05 import fit_gain

    final, losses = fit_gain(2., 0., 5., 0.1, 3)
    assert final == 2.
    assert losses == [25.] * 4


@pytest.mark.parametrize("rate,steps", [(0., 2), (-0.1, 2), (0.1, 0), (0.1, -1)])
def test_gain_rejects_nonpositive_controls(rate, steps):
    from solutions.draft.ch05 import fit_gain

    with pytest.raises(ValueError):
        fit_gain(2., 3., 5., rate, steps)

