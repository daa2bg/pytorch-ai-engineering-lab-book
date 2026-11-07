# Study 23 — Turn predictive error and replay into experiments

**First draft; optional research study.** Biologically inspired terminology is useful only when it leads to an explicit mechanism and a testable prediction. This study examines predictive error, replay, and a toy eligibility trace without claiming that these short programs reproduce the brain or replace established ML methods.

**Predictive coding** describes a family of models in which predictions and mismatches influence inference and learning. A prediction-error signal alone does not establish that an implementation is a predictive-coding network: the architecture, state updates, objective, and learning rule all matter. Our first experiment is simply online linear prediction, which gives us a controlled baseline for investigating error and replay.

## Begin with an observable failure

A scalar model predicts `weight * input`. First it sees examples of one relation, then examples of an opposing relation. Updating on recent data can worsen performance on earlier data. We can measure that loss rather than describe it vaguely as forgetting.

```python
def update(weight, sample, rate=0.1):
    value, target = sample
    error = target - weight * value
    return weight + rate * error * value

def mean_error(weight, samples):
    return sum((target - weight * value) ** 2 for value, target in samples) / len(samples)

old_task = [(1.0, 1.0)] * 12
new_task = [(1.0, -1.0)] * 12
weight = 0.0
for sample in old_task:
    weight = update(weight, sample)
old_error_before = mean_error(weight, old_task)
for sample in new_task:
    weight = update(weight, sample)
old_error_after = mean_error(weight, old_task)
assert old_error_after > old_error_before
print("Earlier-task error:", old_error_before, "->", old_error_after)
```

This example is intentionally conflicting: the same input has two targets and no task identity. One scalar weight cannot satisfy both. Replay cannot repair an impossible representation; it can only change the compromise. That distinction keeps an apparent improvement from being mistaken for a complete memory solution.

## Add replay as one controlled change

**Replay** revisits stored examples during learning. The buffer policy, sampling method, and additional update budget are part of the algorithm. Here we alternate new samples with one old sample and explicitly count the extra work. We keep the model and update rule fixed.

```python
replayed_weight = 0.0
for sample in old_task:
    replayed_weight = update(replayed_weight, sample)
for sample in new_task:
    replayed_weight = update(replayed_weight, sample)
    replayed_weight = update(replayed_weight, old_task[0])
print("No replay old/new:", mean_error(weight, old_task), mean_error(weight, new_task))
print("With replay old/new:", mean_error(replayed_weight, old_task), mean_error(replayed_weight, new_task))
assert mean_error(replayed_weight, old_task) < old_error_after
```

The replay run uses more updates. A fairer follow-up compares equal update budgets and reports both task errors. An **ablation** removes one mechanism while holding the important alternatives fixed. It helps determine whether the mechanism, extra compute, or another change explains a result. No single tiny experiment settles a biological hypothesis.

## A delayed learning signal

An **eligibility trace** keeps a fading record of recent local activity. A later **modulator** can gate or scale its effect on a weight. A toy discrete rule is `trace = decay * trace + activity`, followed by `weight += learning_rate * modulator * trace`. These variables are abstract numbers, not measured synapses or dopamine concentrations.

```python
trace, toy_weight = 0.0, 0.0
for activity, modulator in [(2.0, 0.0), (0.0, 1.0)]:
    trace = 0.5 * trace + activity
    toy_weight += 0.1 * modulator * trace
assert toy_weight == 0.1
print("Delayed gated update:", toy_weight)
```

The first event creates eligibility without changing the weight; the second event uses what remains. A longer delay reduces the contribution under this rule. This is a useful implementation exercise, not a detailed model of the experimental systems discussed in the research literature.

## SOLO exercise — Record the gated updates

Implement `eligibility_updates(events, *, decay=0.5, learning_rate=0.1)`. Accept an iterable of `(activity, modulator)` numeric pairs and return the weight after each event, starting with zero trace and zero weight. Reject decay outside `[0, 1]` or a negative learning rate with `ValueError`; other numeric validity is a caller obligation. Empty input returns an empty list.

```python exercise
def eligibility_updates(events, *, decay=0.5, learning_rate=0.1):
    raise NotImplementedError("Record the delayed gated weight history")

assert eligibility_updates([(2, 0), (0, 1)]) == [0.0, 0.1]
```

**Completion gate:** write a hypothesis, the observation that would refute it, and the controlled baseline. Implement the trace; compare zero and nonzero decay. Optional next experiment: add a task-ID feature and observe whether the earlier contradiction becomes representable. Reference answer: `solutions/draft/ch23.py`; tests: `tests/draft/test_ch23.py`.

Primary reading: [Whittington and Bogacz's predictive-coding network paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5467749/), [Gerstner and colleagues on eligibility traces and three-factor learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/), and [PyTorch's replay-memory implementation in its DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). These sources describe different settings; this study borrows testable mechanisms without treating them as interchangeable theories.
