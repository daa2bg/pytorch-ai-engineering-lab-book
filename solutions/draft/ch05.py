"""Study 05 reference answer: one scalar autograd update."""
import torch


def gradient_step(value, target, learning_rate):
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    parameter = torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
    loss = (parameter - target) ** 2
    loss.backward()
    with torch.no_grad():
        parameter -= learning_rate * parameter.grad
    return parameter.item()


def fit_gain(value, input_value, target, learning_rate, steps):
    """Autograd gain fitting with an initial and post-update loss trace.

    Clearing each step avoids adding derivatives evaluated at different parameter
    values. The graph is rebuilt after each update; no retain_graph is needed.
    For input=3 and target=5, error multiplies by (1 - 18*learning_rate).
    Thus 0.05 contracts fastest of the three exercise rates, 0.1 oscillates while
    shrinking, and 0.2 diverges. This depends on the feature scale and loss.
    """
    if learning_rate <= 0 or steps <= 0:
        raise ValueError("learning_rate and steps must be positive")
    parameter = torch.tensor(float(value), dtype=torch.float64, requires_grad=True)
    losses = []
    for _ in range(steps):
        parameter.grad = None
        loss = (parameter * input_value - target) ** 2
        losses.append(loss.item())
        loss.backward()
        with torch.no_grad():
            parameter -= learning_rate * parameter.grad
    with torch.no_grad():
        losses.append(((parameter * input_value - target) ** 2).item())
    return parameter.item(), losses
