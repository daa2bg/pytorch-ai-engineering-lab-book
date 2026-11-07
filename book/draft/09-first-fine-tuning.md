# Study 09 - Adapt BERT, then decide whether it earns its cost

The SMS router has real inputs and a verified pretrained encoder. It still has no learned connection between those inputs and `ham=0` / `spam=1`. Here we fit that connection, compare two update policies, and save a model that can predict after the notebook is gone. A useful outcome may be that a simpler baseline wins.

You need the minibatch loop and held-out reasoning from Chapters 6–7 and BERT inputs from Chapter 8. The complete CPU experiment uses the same 800 training, 200 validation and 200 test records, batch size 16, a 64-token limit and three epochs per candidate. That is 150 optimizer updates for each BERT policy. Allow about an hour with practice; runtime depends on CPU. Use the repository's `.[book,dev]` environment with PyTorch, Transformers and scikit-learn. No GPU or paid API is required. Worked cells are independent of tagged learner cells.

## Learn a head, or change the representation too?

A **classification head** maps an encoder's representation to our two logits. In `BertForSequenceClassification`, the base's pooled first-token representation passes through dropout and a learned linear classifier. Its output is `[batch, 2]`; integer targets are `[batch]`. Cross-entropy connects the correct class to a gradient, exactly as in the digit classifier. The public base supplies pretrained BERT parameters, but the two-class head starts randomly. Its initialization warning is expected; it is not evidence that the base was randomly initialized.

**Head-only adaptation** holds the encoder fixed and learns a boundary in its representation space. It has fewer trainable parameters and reduces the risk of overwriting useful features. But if SMS distinctions are poorly represented, the head cannot repair them. **Partial fine-tuning** also updates the final encoder layer: gradients can reshape the features while embeddings, the first layer and the pooler remain fixed. More adaptable parameters bring more compute and overfitting risk on just 800 messages. The [BERT model documentation](https://huggingface.co/docs/transformers/model_doc/bert) explains the base and classification outputs; the [training guide](https://huggingface.co/docs/transformers/training) describes supervised adaptation.

Freezing means setting `requires_grad=False` and excluding that tensor from the optimizer. `eval()` is a different switch: it changes dropout behavior but does not prevent gradients. We keep the whole BERT base in evaluation mode for both policies, disabling encoder dropout during this controlled experiment; trainable final-layer weights still receive derivatives. The head's dropout remains active during training. This is a deliberate small-data recipe, not a rule that all fine-tuning must disable encoder dropout.

## Keep data and model identity fixed

Use the same cache policy as Chapter 8. `python tools/prepare_book_assets.py --asset bert --allow-downloads` prepares the verified base. `AI_LAB_CACHE_DIR` chooses the cache root; `AI_LAB_MODEL_DIR` selects a complete base snapshot with provenance. `AI_LAB_SMS_PATH` points to the UCI ZIP/raw file, or `AI_LAB_ALLOW_DOWNLOADS=1` permits first acquisition. Framework loading below is local-only with remote code disabled. `AI_LAB_RUN_PRETRAINED=1` turns missing assets into a hard validation error; it does not permit downloads.

The source is Almeida and Hidalgo (2011), [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI `10.24432/C5CC84`, CC BY 4.0. The base is [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny), MIT, recorded revision `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`. The saved source metadata retains attribution, raw hash and duplicate policy. The deterministic split removes canonical duplicates before selection; it is neither sender-disjoint nor temporal. Test results therefore describe this historical collection, not a deployment guarantee.

```python
import os
import sys
from pathlib import Path
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "config/book_manifest.json").is_file())
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("USE_TF", "0")
from ai_lab.projects.assets import resolve_asset, asset_metadata
from ai_lab.projects.sms import (
    load_sms_records, make_sms_splits, save_sms_splits, load_sms_splits,
    TfidfFeatures, classification_metrics,
)
ARTIFACTS = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", ROOT / "artifacts/projects"))
SPLIT_DIR = ARTIFACTS / "sms-splits-v1"
ready = False
try:
    base_path = resolve_asset("bert")
    base_metadata = asset_metadata(base_path)
    if not SPLIT_DIR.exists():
        save_sms_splits(make_sms_splits(load_sms_records()), SPLIT_DIR)
    splits = load_sms_splits(SPLIT_DIR)
    assert splits.metadata["sizes"] == {"train": 800, "validation": 200, "test": 200}
    ready = True
except FileNotFoundError as error:
    if (os.environ.get("AI_LAB_RUN_PRETRAINED") == "1"
            or os.environ.get("AI_LAB_MODEL_DIR") or os.environ.get("AI_LAB_SMS_PATH")):
        raise
    print("SKIP required adaptation; chapter incomplete:", error)
if ready:
    print("Split SHA:", splits.metadata["split_sha256"])
    print("Class counts:", splits.metadata["class_counts"])
```

The shared loader handles acquisition and saved IDs; the modeling decisions remain visible below. No labels enter the tokenizer. Training alone fits parameters and learned preprocessing. Validation selects candidate and epoch; test is scored only after those choices are frozen. When you attempt the exercise, work above that final test cell and leave it unexecuted until your own choice is fixed. Published worked test results are already known, so further experiments on this public split are development exercises, not new blind tests.

## Establish what a cheap predictor can do

Imbalance makes accuracy deceptive. If 88 of 100 messages are ham, always saying ham gets 88% accuracy while catching no spam. We report a confusion matrix with true classes as rows and predictions as columns, both ordered `[ham, spam]`. A false positive quarantines a legitimate message; a false negative lets spam through. Spam precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`. F1 balances precision and recall; **macro-F1** averages the two class F1 scores equally so ham's larger count cannot dominate. Undefined ratios are reported as zero. This selection metric is educational; an operator's costs may justify a different threshold policy later.

Our second baseline uses training-fitted unigram TF-IDF and logistic regression. TF-IDF weights words by within-message frequency and how uncommon they are among training messages. It ignores word order, but many SMS cues are lexical. The shared implementation uses at most 2,048 terms with training document frequency at least two, log term counts, smoothed IDF and row normalization. Its saved vocabulary/IDF fix the meaning of each column. Logistic regression learns a linear two-class boundary with regularization; unlike BERT it learns no contextual text representation. See [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression) and [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

```python
if ready:
    import torch
    import numpy as np
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    SEED = 42
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False)
    train_text = [r.text for r in splits.train]
    val_text = [r.text for r in splits.validation]
    train_y = [r.label for r in splits.train]
    val_y = [r.label for r in splits.validation]
    majority = Counter(train_y).most_common(1)[0][0]
    features = TfidfFeatures.fit(train_text, max_features=2048, min_df=2)
    baseline = LogisticRegression(max_iter=500, random_state=SEED, C=1.0)
    baseline.fit(features.transform(train_text), train_y)
    baseline_validation = {
        "majority": classification_metrics(val_y, [majority] * len(val_y)),
        "tfidf_logistic": classification_metrics(
            val_y, baseline.predict(features.transform(val_text)).tolist()),
    }
    print("Validation baselines:", baseline_validation)
```

We do not rebalance or choose regularization after inspecting test results. Baselines and BERT see the same records. These two baselines are controls, not obstacles for a larger model to be forced to beat.

## Prepare batches and an auditable update policy

A list of records is a valid map-style dataset for this small run. The loader's `collate_fn` receives a list of selected records and turns it into tensors. This preserves text/label alignment while padding only the current batch. A separate seeded loader generator repeats the same batch order for both candidates.

```python
if ready:
    def collate(records):
        batch = tokenizer([r.text for r in records], padding=True,
                          truncation=True, max_length=64, return_tensors="pt")
        batch["labels"] = torch.tensor([r.label for r in records], dtype=torch.long)
        return batch

    def new_model(policy):
        if policy not in {"head", "last_layer"}:
            raise ValueError("policy must be head or last_layer")
        torch.manual_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=2, id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
            local_files_only=True, trust_remote_code=False)
        for parameter in model.bert.parameters():
            parameter.requires_grad_(False)
        if policy == "last_layer":
            for parameter in model.bert.encoder.layer[-1].parameters():
                parameter.requires_grad_(True)
        return model

    def score(model, records):
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in DataLoader(records, batch_size=16, collate_fn=collate):
                predictions.extend(model(**batch).logits.argmax(1).tolist())
        return classification_metrics([r.label for r in records], predictions), predictions
```

The `[-1]` index chooses the final encoder layer without assuming its numeric name. The head remains trainable because it is outside `model.bert`. Supplying only a parameter-name list would not prove this policy ran: we will compare cloned tensors before and after actual updates.

## Three full epochs, with validation-based checkpoint selection

Use AdamW at 0.001 for the new head and 0.00005 for the final pretrained layer when enabled, with weight decay 0.01. Parameter groups allow different learning rates in one optimizer. The smaller pretrained rate limits each update to an already useful representation; it is a chosen starting recipe, not an optimal value established in advance.

We save the epoch with highest validation macro-F1, breaking ties in favor of the earlier epoch. The copy must clone each tensor: a shallow dictionary copy would still share storage with ongoing training. We train all three epochs to compare their curves, then restore that selected state. No optimizer or RNG state is saved here; this is an inference artifact, not Chapter 7's training-resume checkpoint.

```python
if ready:
    import time
    def fit_policy(policy, *, encoder_lr=5e-5):
        model = new_model(policy)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        head = list(model.classifier.parameters())
        groups = [{"params": head, "lr": 1e-3}]
        if policy == "last_layer":
            groups.append({"params": list(model.bert.encoder.layer[-1].parameters()),
                           "lr": encoder_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
        loader = DataLoader(splits.train, batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED),
                            collate_fn=collate, num_workers=0)
        best_f1, best_state, best_epoch = -1.0, None, None
        history = []
        started = time.perf_counter()
        for epoch in range(1, 4):
            model.train()
            model.bert.eval()  # mode and gradient policy are independent
            total_loss, seen, updates = 0.0, 0, 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                n = len(batch["labels"])
                total_loss += loss.item() * n
                seen += n
                updates += 1
            metrics, _ = score(model, splits.validation)
            history.append({"epoch": epoch, "seen": seen, "updates": updates,
                            "train_loss": total_loss / seen, "validation": metrics})
            if metrics["macro_f1"] > best_f1:
                best_f1, best_epoch = metrics["macro_f1"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        changed = [name for name, p in model.named_parameters()
                   if not torch.equal(before[name], p)]
        frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
        assert not set(changed).intersection(frozen)
        assert any(name.startswith("classifier.") for name in changed)
        if policy == "last_layer":
            assert any(name.startswith("bert.encoder.layer.1.") for name in changed)
        return model, {"policy": policy, "best_epoch": best_epoch, "history": history,
                       "seconds": time.perf_counter() - started,
                       "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                   if p.requires_grad),
                       "changed": changed, "frozen": frozen,
                       "encoder_lr": encoder_lr if policy == "last_layer" else None}

    models, runs, validation = {}, {}, {}
    for policy in ("head", "last_layer"):
        models[policy], runs[policy] = fit_policy(policy)
        validation[policy], _ = score(models[policy], splits.validation)
        print(policy, runs[policy]["trainable_parameters"], "trainable parameters; epoch",
              runs[policy]["best_epoch"], "seconds", round(runs[policy]["seconds"], 2))
        print("Epoch loss/F1:", [(r["epoch"], round(r["train_loss"], 4),
               round(r["validation"]["macro_f1"], 4)) for r in runs[policy]["history"]])
        print("Changed/frozen tensors:", len(runs[policy]["changed"]),
              len(runs[policy]["frozen"]), "validation:", validation[policy])
```

Frozen assertions cover every frozen parameter, including the large embedding table and first layer. Requiring at least one final-layer tensor to change catches a disconnected optimizer group. It would be too strong to require every trainable scalar to change: some gradients may be zero. A falling training loss shows fitting, not necessarily useful generalization. Compare validation macro-F1 and error types before choosing extra capacity.

**Recorded run (2026-10-30):** Python 3.12.9, PyTorch 2.8.0+cpu, Transformers 4.54.1, scikit-learn 1.6.1, two CPU threads. Each policy visited all 800 rows three times (150 updates). Head-only changed 2 tensors and left 39 frozen; final-layer adaptation changed 18 and left 23 frozen. The trainable counts were 258 and 198,530. Measured training/validation/weight-audit time was 1.38s and 1.99s on this AMD64 workstation, excluding model loading; these are observations, not portable timing promises.

Head-only training loss fell from 0.3309 to 0.2000; its selected epoch was 3. Partial fine-tuning fell from 0.2762 to 0.0696, but validation macro-F1 peaked at epoch 2 (0.9116) then fell to 0.9051. Continuing to lower training loss therefore did not select a better validation checkpoint. Both curves and complete confusion matrices are preserved in the saved report.
## Inspect a failure before judging the winner

Choose the higher validation macro-F1, with the smaller head-only policy winning a tie. For each error type, inspect the first ID in saved validation order. The printed diagnostics contain no message text: length/truncation, whether the text contains a currency symbol, and digit count. These are clues for a hypothesis, not a causal explanation. A missed spam message without a currency cue suggests that a simple surface rule cannot recover all spam; it does not prove that BERT relies on that cue. Source text remains in the local prepared dataset for controlled investigation.

```python
if ready:
    selected = ("last_layer" if validation["last_layer"]["macro_f1"] >
                validation["head"]["macro_f1"] else "head")
    chosen = models[selected]
    _, predictions = score(chosen, splits.validation)
    inspected_errors = {}
    for name, truth, prediction in (("false_positive", 0, 1), ("false_negative", 1, 0)):
        indices = [i for i, (row, pred) in enumerate(zip(splits.validation, predictions))
                   if row.label == truth and pred == prediction]
        if not indices:
            inspected_errors[name] = None
            print(name, "none in validation")
            continue
        row = splits.validation[indices[0]]
        n_tokens = len(tokenizer(row.text)["input_ids"])
        inspected_errors[name] = {
            "id": row.record_id, "true": row.label, "predicted": prediction,
            "tokens": n_tokens, "truncated": n_tokens > 64,
            "currency_symbol": any(c in row.text for c in "£$€"),
            "digit_count": sum(c.isdigit() for c in row.text),
        }
        print(name, inspected_errors[name])
    print("BERT choice fixed using validation:", selected)
```

In this run the first false positive was `sms-001361`: 109 tokens, so its input was truncated. The first false negative was `sms-000882`: 36 tokens, no currency symbol and no digits. Neither record is explained by a confident claim about model internals. The first motivates a length-policy investigation; the second shows that restoring removed tokens cannot repair every missed spam message.

If a false negative was truncated, inspect the length policy on validation before blaming model capacity. If it was not truncated, increasing the token limit cannot restore evidence that was already present. A false positive matters even when overall accuracy is high: a correspondence workflow may lose a legitimate request. We postpone abstention thresholds to the serving chapters, where quality and review coverage are evaluated together.

## SOLO - make a controlled adaptation change

Implement `adapt_last_layer(model, train_loader, *, encoder_lr=1e-4, epochs=3)`. The supplied model is a freshly loaded two-label BERT; the reusable CPU loader yields dictionaries containing tokenizer tensors and `labels`. Freeze the base, enable only its last layer, keep the classifier trainable, and use AdamW with head rate 0.001, supplied encoder rate and weight decay 0.01. For each epoch select head training/base evaluation mode, clear gradients, run forward/backward/step and return a list of sample-weighted training losses. Mutate the supplied model deliberately; do not edit batches. Valid positive rates and epoch counts are caller obligations. Raise `ValueError` for a loader yielding no samples.

This is the same training skill as the worked recipe with one controlled rate change. Construct a **fresh** model and fresh seed-42 loader; do not continue the already trained winner. Use `score` on validation after the final epoch, compare with the worked final-epoch result, audit frozen/changed tensors, and explain whether doubling the encoder rate helped. Keep three epochs, batch 16, the split and token limit fixed. The final-epoch comparison avoids giving one policy more checkpoint selection opportunities. A worse outcome is acceptable. Test data is not an input to this function.

```python exercise
def adapt_last_layer(model, train_loader, *, encoder_lr=1e-4, epochs=3):
    raise NotImplementedError("Configure gradients and train the final BERT layer plus head")

candidate = new_model("last_layer")
candidate_loader = DataLoader(
    splits.train, batch_size=16, shuffle=True, collate_fn=collate,
    generator=torch.Generator().manual_seed(SEED), num_workers=0)
frozen_before = candidate.bert.embeddings.word_embeddings.weight.detach().clone()
losses = adapt_last_layer(candidate, candidate_loader)
assert len(losses) == 3
assert torch.equal(frozen_before, candidate.bert.embeddings.word_embeddings.weight)
candidate_validation, _ = score(candidate, splits.validation)
print(losses, candidate_validation)
```

Answer and reasoning: `solutions/draft/ch09.py`; tests: `tests/draft/test_ch09.py`. The answer tests include a full real 800-message, three-epoch run when assets are available. A small fake module can test a loop edge case; it cannot establish that the main adaptation exercise ran.

## Score test once, then save the actual inference bundle

The worked recipe and BERT selection are now fixed. Report both frozen BERT candidates and both baselines on test without using this table to revise their settings. Choosing a production family should also use validation and operational constraints. Saving the selected BERT does not assert it beats TF-IDF or has calibrated probabilities.

```python
if ready:
    test_text = [r.text for r in splits.test]
    test_y = [r.label for r in splits.test]
    final_test = {
        "majority": classification_metrics(test_y, [majority] * len(test_y)),
        "tfidf_logistic": classification_metrics(
            test_y, baseline.predict(features.transform(test_text)).tolist()),
    }
    for policy in ("head", "last_layer"):
        final_test[policy], _ = score(models[policy], splits.test)
    print("Final test, after selection:", final_test)
```

**Observed comparison after the worked selection was fixed:**

| Predictor | Validation macro-F1 | Test accuracy | Test macro-F1 | Test spam precision / recall |
| --- | ---: | ---: | ---: | ---: |
| Majority ham | 0.467 | 0.875 | 0.467 | 0.000 / 0.000 |
| TF-IDF + logistic regression | 0.765 | 0.890 | 0.578 | 1.000 / 0.120 |
| BERT head-only, epoch 3 | 0.806 | 0.945 | 0.851 | 0.938 / 0.600 |
| BERT final-layer + head, epoch 2 | 0.912 | 0.980 | 0.953 | 0.957 / 0.880 |

The selected BERT's validation confusion was `[[170,5],[3,22]]`; its final test confusion was `[[174,1],[3,22]]`. Only 25 spam messages occur in each held-out partition: one changed spam prediction moves recall by four percentage points. TF-IDF's perfect spam precision hides its 22 missed test spam messages. This bounded untuned baseline is a useful warning about imbalance, not proof that lexical models are generally inferior. The BERT advantage here does not establish robustness to new sources, and the small test set should temper deployment claims.
An inference bundle needs weights, architecture, label names, tokenizer, length policy and provenance. We save a new directory, record versions and all source/split metadata, then reload both tokenizer and model locally. Token-ID parity alone would miss lost trained weights; logit parity alone on old token tensors would miss a changed tokenizer. We check both.

```python
if ready:
    import json
    import tempfile
    import platform
    import importlib.metadata
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="sms-bert-v1-", dir=ARTIFACTS))
    chosen.eval()
    chosen.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    report = {
        "task_id": "sms-routing.v1", "label_map": {"ham": 0, "spam": 1},
        "base": base_metadata, "split_metadata": splits.metadata,
        "selected": selected, "selection": "validation macro-F1; head wins tie",
        "tokenization": {"max_length": 64, "padding": "longest", "truncation": True},
        "seed": SEED, "batch_size": 16, "epochs": 3, "threads": torch.get_num_threads(),
        "cpu": platform.processor(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "scikit-learn")},
        "runs": runs, "validation": validation, "baseline_validation": baseline_validation,
        "inspected_errors": inspected_errors, "test": final_test,
    }
    (output_dir / "training-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    restored_tokenizer = AutoTokenizer.from_pretrained(
        output_dir, local_files_only=True, trust_remote_code=False)
    restored_model = AutoModelForSequenceClassification.from_pretrained(
        output_dir, local_files_only=True, trust_remote_code=False).eval()
    probe_text = [r.text for r in splits.validation[:8]]
    old_inputs = tokenizer(probe_text, padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
    new_inputs = restored_tokenizer(probe_text, padding=True, truncation=True,
                                   max_length=64, return_tensors="pt")
    for key in old_inputs:
        assert torch.equal(old_inputs[key], new_inputs[key])
    with torch.no_grad():
        expected = chosen(**old_inputs).logits
        actual = restored_model(**new_inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored_model.config.id2label == {0: "ham", 1: "spam"}
    print("Model AND tokenizer reload passed:", output_dir.name)
```

The output is a real locally restorable classifier, not a reference to notebook globals or a randomly initialized replacement. The shared later serving model may use the lighter TF-IDF/PyTorch path after comparing costs and quality; BERT adaptation remains an executed capability, and Chapter 22 revisits it with low-rank adapters. Chapter 10 now opens the data preparation boundary that made this comparison credible.

**Completion gate:** train both policies for 150 updates each; verify changed and frozen tensors; compare majority, TF-IDF and BERT with the agreed metrics; inspect available validation error types; independently train the rate variant; explain validation selection and the final test boundary; restore model **and** tokenizer with matching logits. Missing-asset skips leave the central outcome incomplete.
