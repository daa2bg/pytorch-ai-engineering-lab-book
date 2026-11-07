# Study 07 - Keep the result: evaluation, checkpoints, and exact resume

The digit recognizer now learns, but a notebook session is not a durable component. A form-processing job needs the same pixel transformation and label meaning tomorrow. A training job interrupted overnight should continue from its last completed epoch instead of quietly restarting its shuffle sequence or forgetting how its optimizer was moving.

This chapter turns Chapter 6's experiment into two artifacts: an **inference bundle** that reconstructs predictions in a new Python process, and a **training checkpoint** that reproduces the next updates. We also make a validation decision before opening the test partition. These are different obligations: restoring predictions does not prove training resumed correctly, and reproducing training does not prove the model generalizes.

**Setup:** use the repository's book environment with PyTorch, NumPy and scikit-learn. All data comes from bundled `load_digits`; no download, GPU or paid API is needed. Run worked cells in order. The complete CPU experiment runs three 30-epoch trajectories (linear, uninterrupted MLP, and 10+20 resumed MLP), typically within a minute. It writes a new run directory under `AI_LAB_ARTIFACT_DIR`, default `artifacts/projects`, and prints its path. Keep that directory to reuse the result. Allow 60-90 minutes with the essential resume exercise.

## Preserve the experiment before improving it

A **split** is part of the experiment, not a convenience to redraw until a score looks better. Training labels drive updates; validation labels guide choices; test labels measure the frozen decision. If the same writer or near-duplicate document appears on both sides, disjoint row IDs alone cannot guarantee independent evaluation. Our loader supplies no writer IDs, so this chapter retains the educational row split and its limitation.

We use Chapter 6's exact seed 7, 1,078/359/360 train/validation/test rows, row-major 64-pixel features, and fixed divisor 16. This measurement-scale constant learns nothing from held-out data. Chapter 4's fitted mean/scale is a different preprocessing choice; silently substituting it would change the recognizer's input contract.

**Data attribution:** E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, [UCI DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49), CC BY 4.0. [scikit-learn's bundled subset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) contains 1,797 images from the original UCI test portion. Our new splits do not reproduce its original writer-separated benchmark.

```python
import hashlib
import json
import os
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
SEED = 7
digits = load_digits()
indices = np.arange(len(digits.target))
train_ids, held_ids = train_test_split(
    indices, test_size=0.4, random_state=SEED, stratify=digits.target)
val_ids, test_ids = train_test_split(
    held_ids, test_size=0.5, random_state=SEED,
    stratify=digits.target[held_ids])
assert (len(train_ids), len(val_ids), len(test_ids)) == (1078, 359, 360)
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
source_ids = np.array([f"digits-{i:06d}" for i in indices])

def tensors(rows):
    return (torch.tensor(digits.data[rows] / 16.0, dtype=torch.float32),
            torch.tensor(digits.target[rows], dtype=torch.long))

x_train, y_train = tensors(train_ids)
x_val, y_val = tensors(val_ids)
# Do not construct or evaluate test tensors during development.
data_contract = {
    "task_id": "digits.v1", "source": "sklearn.datasets.load_digits",
    "source_doi": "10.24432/C50P49", "license": "CC-BY-4.0",
    "source_rows": 1797, "seed": SEED,
    "train_ids": source_ids[train_ids].tolist(),
    "validation_ids": source_ids[val_ids].tolist(),
    "test_ids": source_ids[test_ids].tolist(),
    "scaler": {"kind": "fixed_divisor", "divisor": 16.0, "input_range": [0, 16]},
    "pixel_shape": [8, 8], "feature_order": "row-major", "dtype": "float32",
    "label_to_id": {str(i): i for i in range(10)},
}
source_bytes = digits.data.astype("<f4").tobytes() + digits.target.astype("<i8").tobytes()
data_contract["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
run_id = "ch07-" + uuid.uuid4().hex[:12]
run_dir = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", "artifacts/projects")) / run_id
run_dir.mkdir(parents=True, exist_ok=False)
(run_dir / "data-contract.json").write_text(json.dumps(data_contract, indent=2), encoding="utf-8")
print("Run directory:", run_dir.resolve())
```

The source hash identifies the ordered pixels and targets used here; it is not a quality score. The saved IDs are strings for traceability, never model inputs. A new run directory preserves earlier artifacts instead of overwriting a published model.

## Why optimizer state belongs in a checkpoint

Plain SGD subtracts a scaled gradient. **Momentum** also keeps a running direction from earlier gradients, so the next update depends on both today's gradient and an optimizer buffer. For the default SGD form used here, the buffer follows `buffer = momentum * buffer + gradient`, then the update subtracts `learning_rate * buffer`. Retaining the direction can accelerate progress, but also overshoot. It does not guarantee higher accuracy.

We deliberately add momentum 0.9 and use learning rate 0.05 for this chapter, fixed before looking at results. This is a declared new training policy, not a reproduction of Chapter 6's plain-SGD scores. Both the linear baseline and MLP use this same policy and 30 epochs. Keeping nonempty momentum buffers makes an incomplete resume observable. [PyTorch's SGD documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html) specifies the update and stored state.

```python
config = {
    "kind": "mlp", "input_features": 64, "classes": 10, "hidden": 32,
    "learning_rate": 0.05, "momentum": 0.9, "batch_size": 64, "seed": SEED,
}
TOTAL_EPOCHS = 30
PAUSE_AFTER = 10

def build_model(config):
    if config["kind"] == "linear":
        return nn.Linear(config["input_features"], config["classes"])
    if config["kind"] != "mlp":
        raise ValueError("unsupported model kind")
    return nn.Sequential(
        nn.Linear(config["input_features"], config["hidden"]), nn.ReLU(),
        nn.Linear(config["hidden"], config["classes"]),
    )

def make_run(config):
    torch.manual_seed(config["seed"])
    model = build_model(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"])
    generator = torch.Generator().manual_seed(config["seed"])
    return model, optimizer, generator
```

Model initialization and loader shuffling have separate random-number generators. A **seed** starts a sequence; a **generator state** records the current position in that sequence. Reapplying the seed at epoch 10 would begin the shuffle sequence again. We will save both the loader generator and PyTorch's CPU generator. This particular MLP has no stochastic layers after initialization; preserving the latter still verifies the process state needed when such operations are added.

The training loop remains the one you already know. The returned loss list contains one sample-weighted mean per completed epoch. `start_epoch` counts completed epochs; `total_epochs` is the final budget, not the number of extra epochs to run.

```python
def train_epochs(model, optimizer, generator, features, targets, config,
                 start_epoch, total_epochs):
    """Return post-start epoch mean training losses, including every tail sample."""
    loader = DataLoader(TensorDataset(features, targets),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=generator, num_workers=0, drop_last=False)
    losses = []
    for _ in range(start_epoch, total_epochs):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_y)
            seen += len(batch_y)
        losses.append(total / seen)
    return losses
```

We only checkpoint **between epochs**, after the final minibatch update. Mid-epoch continuation would additionally need the current permutation, position and any partial gradient accumulation. Our `num_workers=0`, CPU-only loop avoids worker-process and accelerator RNG state. Reproducibility across different PyTorch versions or hardware is a separate limitation; see [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

## Interrupt, restore, and compare the next updates

A `state_dict` is a mapping of state names to tensors or optimizer settings. Taking a dictionary is not necessarily an independent snapshot: its tensors can share storage with the live model. We copy state before saving. The checkpoint includes weights, optimizer buffers/settings, completed epochs, prior losses, both RNG positions, architecture and the data contract. [PyTorch's saving guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) distinguishes these from an inference-only weight file.

```python
def save_checkpoint(path, model, optimizer, generator, epoch, losses, config,
                    data_contract):
    """Snapshot all state after a complete epoch; leave no live tensor aliases."""
    checkpoint = {
        "schema_version": 1, "epoch": epoch, "losses": list(losses),
        "config": deepcopy(config), "data_contract": deepcopy(data_contract),
        "model_state": deepcopy(model.state_dict()),
        "optimizer_state": deepcopy(optimizer.state_dict()),
        "torch_rng": torch.get_rng_state().clone(),
        "loader_rng": generator.get_state().clone(),
    }
    torch.save(checkpoint, Path(path))

uninterrupted, full_optimizer, full_generator = make_run(config)
full_losses = train_epochs(uninterrupted, full_optimizer, full_generator,
                           x_train, y_train, config, 0, TOTAL_EPOCHS)
full_torch_rng = torch.get_rng_state().clone()
full_loader_rng = full_generator.get_state().clone()

interrupted, optimizer, generator = make_run(config)
first_losses = train_epochs(interrupted, optimizer, generator,
                            x_train, y_train, config, 0, PAUSE_AFTER)
checkpoint_path = run_dir / "epoch-10.pt"
save_checkpoint(checkpoint_path, interrupted, optimizer, generator,
                PAUSE_AFTER, first_losses, config, data_contract)
assert optimizer.state_dict()["state"]  # Momentum buffers really exist.
```

Now simulate another process's starting point by constructing new objects. Construction itself consumes random numbers, so restore the saved RNG state **after** initialization. Load the optimizer state into an optimizer attached to the new model's parameters, not the old model's parameters.

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
resumed, resumed_optimizer, resumed_generator = make_run(checkpoint["config"])
resumed.load_state_dict(checkpoint["model_state"])
resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
resumed_generator.set_state(checkpoint["loader_rng"])
torch.set_rng_state(checkpoint["torch_rng"])
resumed_losses = checkpoint["losses"] + train_epochs(
    resumed, resumed_optimizer, resumed_generator, x_train, y_train,
    checkpoint["config"], checkpoint["epoch"], TOTAL_EPOCHS)

assert resumed_losses == full_losses
assert all(torch.equal(a, b) for a, b in zip(uninterrupted.parameters(), resumed.parameters()))
assert torch.equal(resumed_generator.get_state(), full_loader_rng)
assert torch.equal(torch.get_rng_state(), full_torch_rng)
for before, after in zip(full_optimizer.state.values(), resumed_optimizer.state.values()):
    assert torch.equal(before["momentum_buffer"], after["momentum_buffer"])
print("30 epochs == 10 saved + 20 resumed: losses, weights, momentum, RNG positions")
```

This equality checks a stronger claim than similar accuracy: the interrupted route arrived at the same numerical training state. It is a scoped CPU test, not a guarantee for arbitrary distributed jobs. `weights_only=True` restricts the loader's supported objects; load these locally produced, trusted artifacts rather than treating that option as authentication.

## Evaluate the choice, not just the loop

The majority training label is a very cheap baseline. A trained linear model is a stronger comparison. The former reveals whether a score comes mostly from class prevalence; the latter asks whether our hidden layer earns its extra complexity under this fixed training policy.

Accuracy counts correct labels. A **confusion matrix** adds error direction: row `i`, column `j` counts true digit `i` predicted as `j`. For a particular digit, **recall** is its diagonal count divided by its row total: how many actual examples were recovered. **Precision** divides the same count by its column total: how many predictions of that digit were correct. Accuracy is reasonable for this roughly balanced ten-class experiment, but the matrix can reveal a weak digit hidden by the average. [The matrix convention](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) is also used by scikit-learn.

```python
def evaluate(model, features, targets):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        guesses = logits.argmax(dim=1)
        loss = nn.functional.cross_entropy(logits, targets).item()
    confusion = torch.bincount(targets * 10 + guesses, minlength=100).reshape(10, 10)
    return {"loss": loss, "accuracy": (guesses == targets).float().mean().item(),
            "confusion": confusion.tolist()}, guesses

linear_config = {**config, "kind": "linear"}
linear, linear_optimizer, linear_generator = make_run(linear_config)
linear_losses = train_epochs(linear, linear_optimizer, linear_generator,
                             x_train, y_train, linear_config, 0, TOTAL_EPOCHS)
validation = {}
validation["linear"], _ = evaluate(linear, x_val, y_val)
validation["mlp"], val_guesses = evaluate(resumed, x_val, y_val)
majority_id = int(torch.bincount(y_train).argmax())
majority_validation = (y_val == majority_id).float().mean().item()
for name, report in validation.items():
    print(name, "validation loss/accuracy", round(report["loss"], 4),
          round(report["accuracy"], 4))
print("Training-majority baseline:", majority_id, round(majority_validation, 4))
selected_name = "mlp" if validation["mlp"]["accuracy"] > validation["linear"]["accuracy"] else "linear"
selected_model = resumed if selected_name == "mlp" else linear
selected_config = config if selected_name == "mlp" else linear_config
print("Frozen choice:", selected_name, "epochs:", TOTAL_EPOCHS)
```

`targets * 10 + guesses` encodes each row/column pair as an index from 0 to 99; `bincount` counts those pairs before reshaping. The feature representation, hyperparameters, epoch budget and tie rule (prefer linear) are now frozen. We did not select an epoch on test results. If you want a different budget or width, use validation and record a new experiment before final assessment.

VALIDATION_RESULT_PENDING

## SOLO - repair a resume and prove what would go wrong

Implement `resume_training(path, train_x, train_y, total_epochs)`, returning `(model, optimizer, generator, losses)`. Read a trusted checkpoint in the format above, reconstruct its configuration, restore all training state, run only the remaining epochs, and return the complete loss list including saved epochs. Reject a final budget smaller than the saved epoch with `ValueError`; equality means no further updates. The caller supplies the same ordered/scaled training rows described by the checkpoint and an integer final budget. Do not mutate those tensors. No validation or test data is needed.

**Starting point:** the worked restoration cell and `train_epochs`. **Hints:** a seed is not the saved shuffle position; call `optimizer.load_state_dict`; set RNG state after construction; use `checkpoint["epoch"]` as the loop start. Write the function independently, then change the interruption point from 10 to 5 and repeat the equality check. A different interruption boundary should not change the final model.

```python exercise
def resume_training(path, train_x, train_y, total_epochs):
    raise NotImplementedError("Restore the next update, not only predictions")

model5, optimizer5, generator5 = make_run(config)
losses5 = train_epochs(model5, optimizer5, generator5, x_train, y_train, config, 0, 5)
path5 = run_dir / "epoch-5-exercise.pt"
save_checkpoint(path5, model5, optimizer5, generator5, 5, losses5, config, data_contract)
repaired, _, _, repaired_losses = resume_training(path5, x_train, y_train, TOTAL_EPOCHS)
assert repaired_losses == full_losses
assert all(torch.equal(a, b) for a, b in zip(uninterrupted.parameters(), repaired.parameters()))
```

Deliberately omit **one** restoration operation in a temporary variant: either optimizer state or loader-generator state. Compare its parameters after one more epoch with a correctly resumed model. Explain why the first predictions immediately after loading can agree while later predictions diverge. The reference tests demonstrate both negative cases; a checkpoint file existing on disk is not enough evidence.

Complete this practice using training/validation data before running the final-test section. Published test results in this chapter are reproduction evidence, not a fresh unseen benchmark for a reader who has already studied them. Reference answers and reasoning: `solutions/draft/ch07.py`; tests: `tests/draft/test_ch07.py`.

## Open the test partition once choices are fixed

We now score the selected model and the already fixed linear baseline, without changing either based on the result. The baseline is reported even if it loses. For a finite test set, differences of a few samples can reflect sampling variation; there is no arbitrary minimum accuracy gate here.

```python
x_test, y_test = tensors(test_ids)
test_report, test_guesses = evaluate(selected_model, x_test, y_test)
linear_test, _ = evaluate(linear, x_test, y_test)
majority_test = (y_test == majority_id).float().mean().item()
print("Selected test loss/accuracy:", round(test_report["loss"], 4),
      round(test_report["accuracy"], 4))
print("Frozen linear/majority test accuracy:", round(linear_test["accuracy"], 4),
      round(majority_test, 4))
confusion = torch.tensor(test_report["confusion"])
assert int(confusion.sum()) == len(y_test)
print("Rows actual, columns predicted; order 0..9:", confusion.tolist())
recalls = confusion.diag().float() / confusion.sum(dim=1)
weakest = int(recalls.argmin())
print("Lowest test recall digit:", weakest, "recall:", round(recalls[weakest].item(), 4))
errors = (test_guesses != y_test).nonzero(as_tuple=True)[0]
for i in errors[:3].tolist():
    print(source_ids[test_ids[i]], "actual", int(y_test[i]), "predicted", int(test_guesses[i]))
```

TEST_RESULT_PENDING

Reading these errors describes the frozen assessment; using them to tune again would spend the test set. This small row-split evaluation still does not establish performance on new writers, new form layouts or different crop preparation. A wrong digit can silently change a reference number, so application-level evaluation must include whole-field correctness as well as single-digit accuracy.

## Save an inference bundle and cross a process boundary

Inference needs weights, architecture, feature order, scaling and label mapping. It does not need momentum buffers. Save these separately from the training checkpoint so a prediction process has a small, explicit dependency boundary. The bundle also records the splits, source identity, training policy and observed metrics. Its file hashes detect accidental changes; they do not establish trust in a bundle from an unknown source.

```python
bundle_dir = run_dir / "inference"
bundle_dir.mkdir()
torch.save(selected_model.state_dict(), bundle_dir / "weights.pt")
# Two validation crops are a restore probe, not a new training or test partition.
probe = {"source_ids": source_ids[val_ids[:2]].tolist(),
         "pixels": digits.data[val_ids[:2]].tolist()}
(bundle_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
manifest = {
    "schema_version": 1, "artifact_id": f"digits-{selected_name}-{run_id}",
    "task_id": "digits.v1", "created_at": datetime.now(timezone.utc).isoformat(),
    "config": selected_config, "epochs": TOTAL_EPOCHS, "data": data_contract,
    "training_policy": "sgd-momentum-v1", "validation": validation[selected_name],
    "test": test_report, "linear_test": linear_test, "majority_test": majority_test,
    "versions": {"torch": str(torch.__version__), "sklearn": sklearn.__version__},
    "files": {name: hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest()
              for name in ("weights.pt", "probe.json")},
}
(bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

The next Python process imports no notebook or answer module and performs no training. It reconstructs the architecture from JSON, verifies the two consumed files, reads the raw probe pixels, applies the saved divisor and produces logits. The small reconstruction code is shown completely so the new process does not depend on a hidden notebook global.

```python
restore_code = r'''
import hashlib, json, sys
from pathlib import Path
import torch
from torch import nn

torch.set_num_threads(1)
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for name in ("weights.pt", "probe.json"):
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != manifest["files"][name]:
        raise ValueError("Artifact checksum mismatch: " + name)
cfg = manifest["config"]
if cfg["kind"] == "linear":
    model = nn.Linear(cfg["input_features"], cfg["classes"])
else:
    model = nn.Sequential(nn.Linear(cfg["input_features"], cfg["hidden"]),
                          nn.ReLU(), nn.Linear(cfg["hidden"], cfg["classes"]))
model.load_state_dict(torch.load(root / "weights.pt", map_location="cpu", weights_only=True))
probe = json.loads((root / "probe.json").read_text(encoding="utf-8"))
scaler = manifest["data"]["scaler"]
if scaler["kind"] != "fixed_divisor":
    raise ValueError("Unsupported preprocessing")
features = torch.tensor(probe["pixels"], dtype=torch.float32) / scaler["divisor"]
model.eval()
with torch.no_grad():
    print(json.dumps(model(features).tolist()))
'''
child = subprocess.run([sys.executable, "-I", "-c", restore_code, str(bundle_dir.resolve())],
                       check=True, capture_output=True, text=True, timeout=60)
restored_logits = torch.tensor(json.loads(child.stdout), dtype=torch.float32)
selected_model.eval()
with torch.no_grad():
    original_logits = selected_model(x_val[:2])
assert torch.equal(restored_logits, original_logits)
print("Fresh-process logits identical; durable bundle:", bundle_dir.resolve())
```

The `-I` option starts Python in isolated mode, excluding notebook-specific Python path settings. The process still uses the selected environment's installed dependencies. Exact logits show that these saved weights and preprocessing reproduce the two probes in this CPU environment; they do not test every possible input. The test suite separately verifies the resume path and deliberately broken state restoration.

**Optional metric extension:** retain the earlier `binary_counts(predictions, targets)` exercise for a two-class task: equally sized binary lists, class 1 positive, reject misalignment/nonbinary entries with `ValueError`, return integer `tp/fp/fn/tn`, and return zeros for empty lists. Explain the zero-denominator policy before deriving precision/recall. This helper alone does not complete this chapter.

**Completion gate:** preserve the approved split and input contract; explain what validation chose before test evaluation; interpret at least one off-diagonal confusion entry; implement the changed-boundary resume and its deliberate failure; and demonstrate a new process reproducing logits from the bundle alone. You now own both a restartable training run and a restorable prediction artifact. The next pretrained-model studies reuse the same train/evaluate/save discipline while adding tokenizers and pretrained weights.
