# Study 05 — Autograd: make one parameter learn

**Outcome:** calculate and check a gradient, predict the effect of an update, and diagnose a training loop that moves the wrong way. PyTorch is required; no dataset or GPU is needed. We keep one scalar visible before the multi-parameter classifier in Chapter 6. Allow 30-45 minutes including the two essential exercises.

Our data pipeline now supplies numerical features. A classifier must learn how strongly those features should affect its scores. We will isolate one weighted feature, `w * x`, so a failed update can be explained with arithmetic rather than hidden inside a network. This synthetic squared-error example teaches the learning mechanism; it is not digit classification or evidence of task quality. A model's **parameters** are numbers that training changes. A **loss** is a scalar measurement of prediction error. To improve a parameter, we need to know how changing it would change that loss. A derivative answers this for one number; a collection of such derivatives is a **gradient**. PyTorch's automatic differentiation, or **autograd**, follows the tensor operations used in a calculation and computes those derivatives.

## Give the parameter a small job

Suppose a one-parameter model predicts `w * x`. For this experiment `x` is three, the desired answer is five, and `w` starts at two. The prediction is six, so its squared error is one.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
prediction = w * 3.0
loss = (prediction - 5.0) ** 2
loss.backward()

assert prediction.item() == 6.0
assert loss.item() == 1.0
assert w.grad.item() == 6.0
print("Prediction, loss, derivative:", prediction.item(), loss.item(), w.grad.item())
```

`requires_grad=True` asks PyTorch to track operations involving this parameter. `backward()` traverses the recorded calculation and accumulates the derivative into `w.grad`. The derivative is positive here: increasing `w` a little would increase the loss. A small step in the opposite direction should help.

For the function `L(w) = (3w - 5)^2`, ordinary calculus gives `L'(w) = 2(3w - 5) * 3`, which is six at `w=2`. You do not need to derive every layer by hand; this small calculation provides a check on what the tool is doing.

## Check a derivative without another differentiation system

A central **finite difference** estimates a slope by evaluating the loss on either side of a point. It is approximate and depends on the step size, but it is a useful independent debugging check.

```python
def scalar_loss(value):
    return (3.0 * value - 5.0) ** 2

epsilon = 1e-5
estimate = (scalar_loss(2.0 + epsilon) - scalar_loss(2.0 - epsilon)) / (2 * epsilon)
assert abs(estimate - w.grad.item()) < 1e-6
print("Finite-difference estimate:", round(estimate, 6))
```

At this scale Python's double-precision float is convenient. Extremely small finite-difference steps can lose accuracy through subtraction; extremely large steps measure behavior too far from the point. Finite differences are an occasional check, not the way we will train large models.

## Update and rebuild the calculation

The **learning rate** controls step size. We move against the derivative, then calculate a fresh prediction and loss. Parameter updates should not themselves become part of the differentiated training calculation, so we put the manual update inside `torch.no_grad()`.

```python
with torch.no_grad():
    w -= 0.1 * w.grad
w.grad = None

next_loss = (w * 3.0 - 5.0) ** 2
next_loss.backward()
assert next_loss.item() < 1.0
print("Updated parameter:", round(w.item(), 3))
print("New loss:", round(next_loss.item(), 3))
```

Gradients accumulate by default. If we kept the old derivative, the next update could mix gradients from separate steps. Assigning `None` clears this scalar's gradient; optimizers provide `zero_grad()` for their parameters. Accumulation can be intentional across several batches, but it must be a deliberate training policy.

We also ran a new forward calculation before the second backward pass. Do not repeatedly call `backward()` on the same saved loss merely to simulate another training step. Training means changing parameters and reevaluating their current predictions. A large learning rate can overshoot; a tiny one can make progress slow.

## Predict whether a step helps

A correct gradient does not guarantee a useful learning rate. For our fixed input `x=3`, define the prediction error as `e = 3w - 5`. The derivative is `6e`, so one update gives `w_new = w - learning_rate * 6e`. Multiplying by 3 and subtracting the target shows that the next error is `(1 - 18 * learning_rate) * e`.

That factor makes a concrete prediction. At rate 0.05 it is 0.1: the error becomes one tenth as large. At 0.1 it is -0.8: the error changes sign but shrinks, so oscillation can still converge. At 0.2 it is -2.6: it overshoots farther each time. This threshold belongs to this particular input and loss, not a universal recommended learning rate. Larger feature scales change the derivative and help explain Chapter 4's interest in normalization.

```python
for rate in (0.05, 0.1, 0.2):
    next_value = 2.0 - rate * 6.0
    next_error = 3.0 * next_value - 5.0
    print("rate", rate, "new w", round(next_value, 3),
          "new loss", round(next_error ** 2, 4))
```

The observed one-step losses are 0.01, 0.64 and 6.76, from an initial loss of 1. The last result is a deliberately disappointing update. Blaming automatic differentiation would miss the actual cause: a step too large for the local curvature.

## Reproduce the stale-gradient mistake

Now keep a reasonable rate, but omit gradient clearing. Predict the second update before running this intentionally broken two-step loop.

```python
buggy = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
trace = []
for step in range(2):
    current_loss = (buggy * 3.0 - 5.0) ** 2
    current_loss.backward()  # BUG: old gradient remains on step 2.
    trace.append((buggy.item(), current_loss.item(), buggy.grad.item()))
    with torch.no_grad():
        buggy -= 0.1 * buggy.grad
print("Before each update (w, loss, accumulated gradient):", trace)
print("Broken final value:", round(buggy.item(), 4))
assert abs(buggy.item() - 1.28) < 1e-10
```

At `w=1.4`, the fresh derivative should be -4.8. Instead, `.grad` contains the old 6 plus -4.8, or +1.2; subtracting it moves the parameter away from the target again. A new forward pass alone does not clear the old gradient buffer. The fix is to reset that buffer for each independent update. The next exercise requires implementing and testing that repair across multiple steps.

## Exercise — Implement one visible learning step

Write `gradient_step(value, target, learning_rate)`. Create a scalar floating-point tensor from `value` with gradient tracking. For this exercise the prediction is the parameter itself and the loss is `(parameter - target) ** 2`. Differentiate, take one no-grad step, and return the updated value as a Python float. Require a positive learning rate; reject zero/negative rates with `ValueError`. The caller supplies finite numerical arguments.

```python exercise
def gradient_step(value, target, learning_rate):
    raise NotImplementedError("Differentiate and update one scalar")

updated = gradient_step(3.0, 1.0, 0.1)
assert abs(updated - 2.6) < 1e-6
assert (updated - 1.0) ** 2 < (3.0 - 1.0) ** 2
```

## SOLO - repair the loop and choose a rate

Implement `fit_gain(value, input_value, target, learning_rate, steps)`. Keep one float64 scalar parameter across the run. At each step, rebuild the squared loss `(parameter * input_value - target) ** 2`, clear stale gradients, differentiate and update without tracking the update operation. Return `(final_value, losses)`, where `losses` contains the loss **before any update and after each update** (length `steps + 1`), all Python floats. Reject a nonpositive learning rate or nonpositive number of steps with `ValueError`; the caller supplies finite scalar arguments and an integer step count. No tensor input is mutated.

**Hints:** clearing before each `backward()` is sufficient. Use a new forward calculation after updates when recording the final loss. Use `.item()` rather than storing graph-connected tensors in the history. You may use the worked scalar arithmetic to predict expected results, but your implementation must use autograd.

```python exercise
def fit_gain(value, input_value, target, learning_rate, steps):
    raise NotImplementedError("Repair the repeated autograd update")

final_value, losses = fit_gain(2.0, 3.0, 5.0, 0.1, 2)
assert abs(final_value - 1.88) < 1e-10
assert len(losses) == 3
assert abs(losses[-1] - 0.4096) < 1e-10
for rate in (0.05, 0.1, 0.2):
    final_value, losses = fit_gain(2.0, 3.0, 5.0, rate, 8)
    print(rate, losses[0], losses[-1])
```

Before executing, classify each rate as shrinking steadily, shrinking with oscillation, or diverging. Use the observed losses to select one rate for this fixture and explain your decision. Then remove gradient clearing in a temporary copy and show which two-step assertion catches the bug. A decreasing loss is useful evidence, but matching the predicted two-step result is a stronger check of this exact update policy.

**Optional study:** repeat with `input_value=6`. Recompute the error multiplier before choosing a rate; do not assume the same setting remains suitable.

**Completion gate:** compare a derivative with the finite-difference estimate, implement both exercises, predict and verify the three learning-rate outcomes, and show that your repaired loop rejects the stale-gradient trajectory. Next we let `nn.Module` hold multiple parameters and an optimizer update them together.

References: [automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [optimization](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html). Answer: `solutions/draft/ch05.py`; tests: `tests/draft/test_ch05.py`.
