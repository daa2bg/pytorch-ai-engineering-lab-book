# Study 06 - Train a classifier that reads handwritten digits

Our correspondence application receives scanned forms as well as text. One useful component would read an isolated handwritten digit in a reference-number box. The input to this component is an already cropped, normalized image; its output is a digit from 0 to 9. Finding boxes on a page and recognizing a whole number are separate jobs. This chapter trains the small recognition component so we can inspect what learning actually changes.

You already know how to align features and targets, collect batches, and calculate gradients. We will first make four easy points learn, then train a linear baseline and a small neural network on real handwritten digits. The result is a reproducible comparison with inspected mistakes, not a claim that we can process arbitrary scans.

**Setup:** use a Python environment with PyTorch, NumPy, scikit-learn and matplotlib. From the repository root, `python -m pip install -e ".[book,dev]" matplotlib` supplies the book dependencies and plotting support if needed. Select that environment as the notebook kernel. The data ships with scikit-learn; no dataset download, GPU or paid service is required. The bounded CPU experiment takes seconds to a minute on typical development machines. Run worked cells in order; cells tagged `exercise` are intentionally unfinished. Allow 60-90 minutes with practice.

## From a category to a learning signal

A **classifier** chooses a category. Digit IDs are names encoded as integers: predicting 8 when the answer is 9 is not automatically a smaller mistake than predicting 1. We therefore produce one score per class instead of fitting a single numerical answer.

Those scores are **logits**, unrestricted numbers whose relative sizes matter. Softmax converts them to nonnegative probabilities summing to one. For two logits `[0, 0]`, the probabilities are `[0.5, 0.5]`; increasing the correct class's logit raises its assigned probability. Classification chooses the largest score, which is also the largest softmax probability.

Accuracy counts correct choices, but a small score change often leaves the winning class unchanged. We train with a smoother **loss**: cross-entropy is the negative natural logarithm of the probability assigned to the correct class. Correct-class probabilities of 0.5 and 0.9 give losses about 0.693 and 0.105. Confident wrong answers cost more. PyTorch's [cross-entropy implementation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html) combines the necessary calculations stably: pass raw logits, not probabilities. Here its input is `[batch, classes]` and its target is a `[batch]` tensor of integer class IDs.

An **optimizer** turns gradients into parameter updates. Stochastic gradient descent (SGD) subtracts the learning rate times each parameter's gradient. The learning rate sets the step size; too small may learn slowly, while too large may overshoot. An **epoch** is one traversal of the training data. Our first example has one batch per epoch, so each epoch makes one update.

```python
import torch
from torch import nn

torch.set_num_threads(1)
torch.manual_seed(7)
tiny_x = torch.tensor([[-2., -1.], [-1., -2.], [1., 2.], [2., 1.]])
tiny_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
tiny_model = nn.Linear(2, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.2)
history = []

tiny_model.train()
for epoch in range(40):
    optimizer.zero_grad(set_to_none=True)
    logits = tiny_model(tiny_x)
    loss = loss_fn(logits, tiny_y)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

assert logits.shape == (4, 2)
assert history[-1] < history[0]
print("First/last fixture loss:", round(history[0], 4), round(history[-1], 4))
```

`nn.Linear(2, 2)` learns a weighted sum plus a bias for each output. `model.parameters()` exposes the tensors the optimizer must update. Each iteration clears old gradients, calculates fresh scores and loss, computes derivatives with `backward()`, and changes parameters with `step()`. These last two operations are different: gradients can exist while weights remain unchanged. Chapter 5's accumulation behavior explains why we clear gradients before each ordinary update. See the [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

`.item()` saves a Python number, rather than retaining a loss tensor and its computation history. `train()` selects training behavior; `eval()` selects evaluation behavior for layers such as dropout. Neither turns gradient tracking on or off. Evaluation below separately uses `no_grad()` because it needs predictions, not derivatives.

```python
tiny_model.eval()
with torch.no_grad():
    tiny_predictions = tiny_model(tiny_x).argmax(dim=1)
assert torch.equal(tiny_predictions, tiny_y)
print("Training-fixture predictions:", tiny_predictions.tolist())
```

`argmax(dim=1)` chooses a class within each row. Four correct training answers check the mechanism, not generalization. The deliberately easy fixture has now done its job; we need examples whose answers are not embedded in one obvious rule.

## What a hidden layer adds

A linear model gives each input pixel a fixed contribution to each digit score. The boundary between any pair of classes is where their two weighted sums are equal: a straight line with two inputs, a flat hyperplane with more. That can be a strong baseline, but it cannot express every relationship.

Consider two binary measurements. Suppose class 1 means **exactly one** is on: `(0, 1)` and `(1, 0)`; class 0 means `(0, 0)` or `(1, 1)`. This is XOR. The midpoint of each class's two points is `(0.5, 0.5)`, so one straight boundary cannot put the two classes on opposite sides. More training cannot fix this representational limit.

A hidden layer first computes intermediate features. An **activation** changes them nonlinearly; ReLU keeps positive values and replaces negative values with zero. With `s = x1 + x2`, the feature `ReLU(s) - 2 * ReLU(s - 1)` is 0, 1, 1, 0 on those four inputs. A final threshold can separate them. This is an explanation with chosen weights, not a trained XOR benchmark. In training, gradients learn such intermediate features rather than our hand-writing them.

Without an activation, two affine layers collapse into one: multiplying weighted sums still produces a weighted sum plus a bias. A **multilayer perceptron (MLP)** inserts an activation between learned layers. Its hidden units may learn useful combinations of pixels, but we should not assume each unit has a clean meaning such as "loop" or "stroke." Extra capacity can help, or fit peculiarities of the training examples. Held-out results must decide. PyTorch's [model walkthrough](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) introduces these components.

## Real digits, with a protected evaluation boundary

We use the 1,797 images bundled in [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Each is an 8x8 grid with values from 0 to 16. Flattening preserves a fixed row-major order of 64 measurements. This is a small real handwriting dataset, although scikit-learn groups it among its toy datasets because of its size.

**Attribution:** E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, [UCI repository, DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), licensed CC BY 4.0. The bundled subset is the original UCI test portion. We make new educational splits within that subset; we do not reproduce the original writer-separated benchmark.

Training rows drive gradient updates. **Validation** rows help compare model choices. **Test** rows are reserved for the final assessment after choices are frozen. Looking at test mistakes while choosing width or epochs would make test results part of development. Here we create test IDs but never score that partition; Chapter 7 develops evaluation and checkpoints further.

A **stratified** split approximately preserves each digit's frequency. Row-level random splitting answers "how well do we recognize other examples from this collection?" It does not establish performance on new writers or new scanners. This loader does not give us writer IDs for a writer-group split. Real form deployment needs representative crops and an appropriate grouping policy.

```python
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 7
digits = load_digits()
row_ids = np.arange(len(digits.target))
source_ids = np.array([f"digits-{i:06d}" for i in row_ids])
train_ids, held_ids = train_test_split(
    row_ids, test_size=0.4, random_state=SEED, stratify=digits.target
)
val_ids, test_ids = train_test_split(
    held_ids, test_size=0.5, random_state=SEED,
    stratify=digits.target[held_ids],
)
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
assert len(train_ids) + len(val_ids) + len(test_ids) == len(row_ids)

# The scale comes from the documented measurement range, not a fitted statistic.
x_train = torch.tensor(digits.data[train_ids] / 16.0, dtype=torch.float32)
y_train = torch.tensor(digits.target[train_ids], dtype=torch.long)
x_val = torch.tensor(digits.data[val_ids] / 16.0, dtype=torch.float32)
y_val = torch.tensor(digits.target[val_ids], dtype=torch.long)
print("Train/validation/test:", len(train_ids), len(val_ids), len(test_ids))
print("Training class counts:", torch.bincount(y_train).tolist())
```

Dividing by 16 brings all features into `[0, 1]` without estimating anything from held-out examples. This differs from Chapter 4's fitted mean and standard deviation: if we chose that transformation, we would fit it on training rows only and reuse the stored values. Never normalize every inference batch independently. [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) uses `random_state` to reproduce the row assignment; seeding does not promise bitwise agreement across all library versions and hardware.

Before training, look at a training example. A form-processing input must preserve the same crop, pixel scale and feature order; an arbitrary photograph resized to 8x8 does not automatically meet that contract.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.imshow(digits.images[train_ids[0]], cmap="gray_r", vmin=0, vmax=16)
ax.set_title(f"Training row {train_ids[0]}: digit {y_train[0].item()}")
ax.axis("off")
plt.show()
plt.close(fig)
```

## Batches make the small loop reusable

A **minibatch** is a subset used for one update. Its gradient estimates an update direction from fewer examples, so it varies across batches. Shuffling changes which examples share an update and avoids repeatedly following one label order. An epoch visits every training example once here, including the incomplete final batch.

A `TensorDataset` supplies the aligned `(features, target)` pair at each index, the same access contract as Chapter 3's indexable dataset. A `DataLoader` assembles those samples into batches. Unlike a one-shot iterator, the loader can produce a fresh traversal each epoch, connecting Chapter 1's iterator lesson to an actual trainer. We use `num_workers=0` for a simple local process. See [PyTorch data loading](https://docs.pytorch.org/docs/2.8/data.html).

```python
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(x_train, y_train)

def make_train_loader(batch_size=64):
    return DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0, drop_last=False,
    )

preview_loader = make_train_loader()
batch_x, batch_y = next(iter(preview_loader))
assert batch_x.shape == (64, 64)
assert batch_y.shape == (64,)
print("Batches per epoch:", len(preview_loader))
```

The first 64 in the shape counts samples; the second counts pixels. Using a separate seeded generator for the loader lets us reset its shuffle sequence for each candidate without depending on model initialization consuming random numbers.

Our MLP maps 64 pixels to 32 hidden activations, then to ten logits. `nn.Module` tracks trainable submodules; `super().__init__()` initializes that machinery. `nn.Sequential` calls its layers in order. `forward` states the calculation, and `model(features)` invokes it through the module interface. Default randomly initialized weights give hidden units different starting points; starting every unit identically can leave them learning the same feature.

```python
class DigitMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, 10)
        )

    def forward(self, features):
        return self.layers(features)

assert DigitMLP()(batch_x).shape == (64, 10)

def evaluate(model, features, targets):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        loss = loss_fn(logits, targets).item()
        accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return {"loss": loss, "accuracy": accuracy}
```

We choose 30 epochs and learning rate 0.2 before comparing candidates. Both see the same split, scale, minibatch sequence, and number of updates. This is a controlled small experiment, not a fully tuned competition: another optimizer or budget may change their ranking. The linear baseline has fewer parameters and is worth keeping if extra complexity brings little benefit.

```python
def fit(model, epochs=30):
    loader = make_train_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    records = []
    for epoch in range(epochs):
        model.train()
        loss_total, seen = 0.0, 0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(targets)
            seen += len(targets)
        validation = evaluate(model, x_val, y_val)
        records.append({
            "epoch": epoch + 1, "train_loss": loss_total / seen,
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
        })
    return records

models, histories, results = {}, {}, {}
for name in ("linear", "mlp"):
    torch.manual_seed(SEED)
    model = nn.Linear(64, 10) if name == "linear" else DigitMLP(32)
    records = fit(model)
    models[name], histories[name] = model, records
    results[name] = evaluate(model, x_val, y_val)
    parameters = sum(p.numel() for p in model.parameters())
    print(name, "parameters", parameters,
          "first/last train loss", round(records[0]["train_loss"], 4),
          round(records[-1]["train_loss"], 4),
          "validation", {k: round(v, 4) for k, v in results[name].items()})
```

Multiplying each mean batch loss by its sample count gives the final short batch its proper weight. An unweighted average of batch means would treat a 54-example batch as if it contained 64. Training loss is measured during updates; validation loss uses one fixed model after the epoch, so these are not exactly the same measurement procedure.

## Read the result before changing the model

Accuracy is a useful first metric because these ten classes have similar counts and our initial task treats a wrong digit equally regardless of its identity. It cannot show which digit pairs cause trouble. Loss additionally distinguishes tentative correct answers from confidently wrong ones.

**Recorded CPU run:** Python 3.12.9, PyTorch 2.8.0+cpu and scikit-learn 1.6.1, seed 7, 1,078 training rows and 359 validation rows:

| Model | Parameters | First / last training loss | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| Linear | 650 | 2.0301 / 0.2768 | 0.2675 | 96.66% (347/359) |
| MLP, 32 hidden units | 2,410 | 2.2416 / 0.1108 | 0.1332 | 97.49% (350/359) |

The MLP gets three more validation digits correct with about 3.7 times as many parameters. That is useful evidence for this run, not proof that nonlinear models always win or that three examples establish a reliable advantage. These are observations for this configuration, not guaranteed scores or a requirement that the MLP win. A change in runtime versions or seed may change them.

```python
fig, ax = plt.subplots(figsize=(6, 3))
for name, records in histories.items():
    ax.plot([r["epoch"] for r in records],
            [r["train_loss"] for r in records], label=f"{name}: train")
    ax.plot([r["epoch"] for r in records],
            [r["val_loss"] for r in records], "--", label=f"{name}: validation")
ax.set(xlabel="Epoch", ylabel="Cross-entropy")
ax.legend()
fig.tight_layout()
plt.show()
plt.close(fig)
```

Falling training loss means the model is fitting seen data. If validation loss rises while training loss falls, that is evidence of **overfitting**: further fitting is no longer helping unseen examples under this split. It is not a signal to inspect the test set. Here we keep the preselected epoch budget; later we can develop an explicit validation-based stopping policy.

Choose the higher validation accuracy, with the simpler linear model winning a tie. Then inspect the first three validation errors in the fixed partition order, rather than choosing the most attractive examples.

```python
selected_name = "mlp" if results["mlp"]["accuracy"] > results["linear"]["accuracy"] else "linear"
selected_model = models[selected_name]
selected_model.eval()
with torch.no_grad():
    val_predictions = selected_model(x_val).argmax(dim=1)
wrong = (val_predictions != y_val).nonzero(as_tuple=True)[0]
print("Selected:", selected_name, "validation mistakes:", len(wrong))
if len(wrong):
    shown = wrong[:3].tolist()
    fig, axes = plt.subplots(1, len(shown), figsize=(3 * len(shown), 3), squeeze=False)
    for ax, index in zip(axes[0], shown):
        row_id = int(val_ids[index])
        truth, prediction = int(y_val[index]), int(val_predictions[index])
        print(source_ids[row_id], "true", truth, "predicted", prediction)
        ax.imshow(digits.images[row_id], cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Row {row_id}: {truth} -> {prediction}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)
```

The recorded MLP made nine validation mistakes. The first three were source rows `digits-001038` (9 predicted as 3), `digits-001765` (3 as 5), and `digits-000678` (5 as 4). In the first crop, the upper loop of the 9 is poorly separated at this resolution; its dark right edge and lower curve can resemble a 3. This visual observation suggests a representation problem worth investigating, but does not prove why the model erred. The other two show that errors are not confined to that digit pair. The displayed learning curves fall for both models across this budget, with no sustained rise in validation loss; that still does not establish performance on new scans. An incorrect digit can alter a reference number while still producing a plausible-looking string. Before using this component in forms, collect errors from representative crops and check whether image preparation preserves distinguishing strokes. This dataset experiment establishes a training workflow; it does not validate that new input distribution.

## SOLO - repair, train, and make a decision

### 1. Turn a broken update into a reusable training step

The function below calculates gradients but omits two essential operations. Implement `training_step(model, optimizer, features, targets)` so it selects training mode, clears old optimizer gradients, computes cross-entropy on logits, backpropagates, updates weights, and returns the **pre-update** loss as a Python float. The caller supplies aligned float features and long class IDs; rely on PyTorch for malformed tensor errors. Mutating model parameters and optimizer state is intentional; do not modify input tensors.

Before fixing it, predict what repeated calls do to the parameters and gradients. Then compare a cloned parameter before and after your repaired call. A nonempty `.grad` alone is not proof of learning. **Hint:** the missing operations belong before the forward pass and after `backward()`.

```python exercise
def training_step(model, optimizer, features, targets):
    model.train()
    loss = nn.functional.cross_entropy(model(features), targets)
    loss.backward()
    # SOLO: repair the missing operations.
    raise NotImplementedError("Finish the update and return its pre-update loss")

exercise_model = nn.Linear(2, 2)
exercise_optimizer = torch.optim.SGD(exercise_model.parameters(), lr=0.2)
before = exercise_model.weight.detach().clone()
first_loss = training_step(exercise_model, exercise_optimizer, tiny_x, tiny_y)
assert isinstance(first_loss, float)
assert not torch.equal(before, exercise_model.weight)
```

### 2. Train a smaller candidate without consulting test data

Implement `train_candidate(train_x, train_y, val_x, val_y, *, hidden=16, epochs=30, batch_size=64, seed=7)`. Return `(model, records)`, where `model` has architecture `64 -> hidden -> ReLU -> 10`, and `records` contains one dictionary per epoch with keys `epoch`, `train_loss`, `val_loss`, and `val_accuracy` as in `fit`.

Seed initialization and loader shuffling separately, use SGD with learning rate 0.2, and retain the final partial batch. Use your repaired `training_step` for each update and the worked `evaluate` for validation. All tensors are nonempty CPU tensors with the preceding shape/dtype contract; the three size arguments are positive integers. Input validation beyond that caller contract is not required. The function creates and trains a fresh model, leaves source tensors unchanged, and never needs test inputs.

**Hints:** build a `TensorDataset` from the supplied training tensors, not global variables; construct the loader once and iterate it afresh each epoch; weight mean batch losses by batch size. Copy the structure of `fit`, then change the width independently.

```python exercise
def train_candidate(train_x, train_y, val_x, val_y, *,
                    hidden=16, epochs=30, batch_size=64, seed=7):
    raise NotImplementedError("Train and evaluate a new MLP using the supplied tensors")

candidate, candidate_records = train_candidate(x_train, y_train, x_val, y_val)
assert len(candidate_records) == 30
assert candidate(x_val[:5]).shape == (5, 10)
assert candidate_records[-1]["train_loss"] < candidate_records[0]["train_loss"]
print(candidate_records[-1])
```

Record the smaller model's parameter count, final validation accuracy, loss, and at least one error ID. Compare it with both worked candidates. Does a width change help enough to justify its cost? A worse score is a valid outcome; explain it without changing the split or silently increasing the budget. The loss-decrease check is a basic learning check for this exercise, not evidence that one architecture must beat another. Keep the test partition untouched while doing this practice.

Reference answers and reasoning are in `solutions/draft/ch06.py`; tests are in `tests/draft/test_ch06.py`. Compare only after your own implementation and explanation.

## What we carry forward

The reusable result is a trained model plus its input contract and evaluation decision. Keep those together: weights without pixel order, scale, digit IDs, and architecture do not define a usable recognizer. The following in-memory metadata records the worked experiment; it does not yet save or reload anything. Source IDs such as `digits-001038` preserve the zero-based row identity in this bundled dataset; they are metadata, never model features.

```python
run_contract = {
    "task_id": "digits.v1", "dataset": "sklearn.datasets.load_digits",
    "dataset_rows": len(row_ids), "source_doi": "10.24432/C50P49",
    "source_license": "CC-BY-4.0",
    "seed": SEED, "split_policy": "stratified rows, approximately 60/20/20",
    "train_ids": source_ids[train_ids].tolist(),
    "val_ids": source_ids[val_ids].tolist(),
    "test_ids": source_ids[test_ids].tolist(), "pixel_shape": [8, 8],
    "feature_order": "row-major", "pixel_divisor": 16.0,
    "label_to_id": {str(i): i for i in range(10)}, "model": selected_name,
    "hidden": 32 if selected_name == "mlp" else None,
    "epochs": 30, "batch_size": 64, "optimizer": "SGD", "learning_rate": 0.2,
    "validation": results[selected_name], "test_evaluated": False,
    "torch_version": str(torch.__version__), "sklearn_version": sklearn.__version__,
}
print("Handoff:", run_contract["model"], "test evaluated:", run_contract["test_evaluated"])
```

Chapter 7 develops held-out evaluation and persisted checkpoints, including the difference between restoring inference and resuming training. Digit labels here are separate from the correspondence categories in the earlier Python exercises; this optional form component must not redefine their label mapping.

**Completion gate:** explain why logits and integer class targets fit this task; explain what ReLU adds that a second linear layer cannot; demonstrate a real parameter update and stale-gradient repair; train the 16-unit candidate with every training row; and justify a model choice using validation evidence and an inspected mistake. Report the reserved test boundary and the limits of this row-level split. Optional GPU work or larger architectures do not block this gate.
