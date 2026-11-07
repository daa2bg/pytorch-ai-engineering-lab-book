# Study 08 - Give a pretrained model the inputs it expects

Our correspondence router now needs to read text. The digit model from Chapters 6–7 cannot consume a message, and the early `claim`/`invoice` examples were synthetic Python fixtures. From here the real teaching task is `sms-routing.v1`: classify public SMS messages as legitimate (`ham=0`) or unwanted (`spam=1`). An operator may later send uncertain predictions to `review`; that is an operational route, not a third training label.

This chapter builds the input side of that system. We will run a real pretrained tokenizer and encoder, inspect the resulting shapes, measure how much text a length limit removes, and save the tokenizer with its provenance. Chapter 9 adds a classifier and learns from labels. No classification quality is implied by obtaining a vector.

**Starting point:** tensors, batches, evaluation mode and saved artifacts from Chapters 4–7. Use the repository's CPU book environment (`python -m pip install -e ".[book,dev]"`) as your notebook kernel. Allow 45–60 minutes with practice; the worked cached-model run is small. Run worked cells in order and leave tagged exercises unexecuted until you attempt them. No paid service or GPU is required.

## A token ID is an address; a representation is a learned vector

A tokenizer splits text into vocabulary entries and returns their integer IDs. BERT uses word pieces: a rare word can become several familiar fragments. Special entries mark sequence boundaries and padding. The tokenizer's normalization and vocabulary must match the encoder's training. Replacing it with `text.split()` would change the meaning of every input address.

An embedding table looks up a vector for each ID. ID 20 is not twice as positive, long, or meaningful as ID 10. The encoder then mixes information across positions, so a token's final **contextual representation** depends on surrounding tokens. The same word in two sentences can have the same ID but different output vectors. Pretraining learned these parameters before our SMS task; it did not learn our two-class head. The [BERT paper](https://arxiv.org/abs/1810.04805) describes masked-word pretraining and downstream adaptation. We need its input contract now; implementing attention from scratch can wait until the architecture study.

Here is a deliberately synthetic vocabulary that makes the addressing visible. It is not BERT's vocabulary.

```python
vocabulary = {"[PAD]": 0, "[UNK]": 1, "message": 2, "received": 3}
sequences = [[2, 3], [2], [1, 1]]
width = max(map(len, sequences))
fixture_ids = [row + [0] * (width - len(row)) for row in sequences]
fixture_mask = [[1] * len(row) + [0] * (width - len(row)) for row in sequences]
assert fixture_ids == [[2, 3], [2, 0], [1, 1]]
assert fixture_mask == [[1, 1], [1, 0], [1, 1]]
print("IDs:", fixture_ids, "mask:", fixture_mask)
```

**Padding** fills short rows so a batch is rectangular. The **attention mask** marks real positions with 1 and padding with 0. Unknown vocabulary entries are still real input; they get mask 1. **Truncation** removes tokens beyond a limit, possibly removing useful evidence. Padding adds storage; truncation loses information. For single BERT messages our limit includes `[CLS]` and `[SEP]`, leaving 62 content positions at a limit of 64. See the [padding/truncation guide](https://huggingface.co/docs/transformers/main/pad_truncation).

## Acquire once, then load verified local files

We use [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny), a converted compact BERT with two encoder layers and hidden width 128; its model card declares MIT. Our recorded snapshot revision is `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`. This small base is useful for CPU learning, not a promise of state-of-the-art SMS accuracy.

Downloads default to disabled. From the repository root, explicitly acquire the base with `python tools/prepare_book_assets.py --asset bert --allow-downloads`. The helper bounds downloads, records immutable revision/file hashes and verifies the cache. `AI_LAB_CACHE_DIR` chooses the cache root; `AI_LAB_MODEL_DIR` can identify a complete BERT snapshot with its `metadata.json`. `AI_LAB_ALLOW_DOWNLOADS=1` also permits the helper to acquire a missing asset. Framework calls below always read the resolved local path and disable custom remote code.

The dataset is Almeida and Hidalgo's (2011) [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI `10.24432/C5CC84`, CC BY 4.0. Set `AI_LAB_SMS_PATH` to its source ZIP or `SMSSpamCollection` file, or enable downloads for first preparation. The shared loader preserves original text and physical-line IDs, removes canonical duplicates, and excludes conflicting-label groups. It creates the frozen seed-42 800/200/200 partitions once; later chapters reload their saved IDs. Chapter 10 explains those decisions. This historical English SMS collection is a proxy for correspondence, not proof of performance on current email or new senders.

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
    ready = True
except FileNotFoundError as error:
    # A supplied but invalid location is a configuration error, never a fallback.
    if (os.environ.get("AI_LAB_RUN_PRETRAINED") == "1"
            or os.environ.get("AI_LAB_MODEL_DIR") or os.environ.get("AI_LAB_SMS_PATH")):
        raise
    print("SKIP required pretrained experiment; chapter incomplete:", error)
if ready:
    assert splits.metadata["sizes"] == {"train": 800, "validation": 200, "test": 200}
    print("Base revision:", base_metadata["revision"])
    print("Split identity:", splits.metadata["split_sha256"])
    print("Class counts:", splits.metadata["class_counts"])
```

`next(...)` finds the repository while allowing the notebook to start in its own directory. The shared helper is project preparation code, not a hidden answer: it supplies checked records and split membership, while the tokenizer and model work remains here. Missing required assets print an explicit incomplete result. Set `AI_LAB_RUN_PRETRAINED=1` for validation that must fail instead of skipping. That switch does not authorize downloads. A corrupt cache raises an error; it is not silently replaced with random weights.

## Read what the actual tokenizer produces

`AutoTokenizer` chooses a tokenizer from checkpoint configuration. `return_tensors="pt"` requests PyTorch tensors. A returned dictionary can be expanded with `**encoded`, equivalent to supplying `input_ids=...`, `attention_mask=...` and other named arguments. `token_type_ids` identifies sentence segments; our single-message inputs use segment 0 throughout.

```python
if ready:
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(
        base_path, local_files_only=True, trust_remote_code=False)
    encoder.eval()
    # Invented safe sentences expose tokens without logging public messages.
    probe_texts = ["A message arrived.", "An unexpectedly long message arrived today."]
    probe = tokenizer(probe_texts, padding=True, truncation=True,
                      max_length=64, return_tensors="pt")
    print("Probe tokens:", tokenizer.convert_ids_to_tokens(probe["input_ids"][0]))
    print("Input shapes:", {k: tuple(v.shape) for k, v in probe.items()})
    print("Mask:", probe["attention_mask"].tolist())
    assert probe["input_ids"].dtype == torch.long
    assert encoder.config.hidden_size == 128
    assert encoder.config.num_hidden_layers == 2
```

Now run original text from the first four **training** records. We print only IDs and shapes. Tokenization cannot learn from labels, but developing input policies by looking through the test corpus would still weaken our held-out boundary.

```python
if ready:
    messages = [row.text for row in splits.train[:4]]
    encoded = tokenizer(messages, padding=True, truncation=True,
                        max_length=64, return_tensors="pt")
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
    assert hidden.shape == (*encoded["input_ids"].shape, 128)
    print("Record IDs:", [row.record_id for row in splits.train[:4]])
    print("Tokens -> contextual vectors:", tuple(encoded["input_ids"].shape),
          "->", tuple(hidden.shape))
```

The shape is `[batch, positions, hidden_width]`. Each position now has 128 floating-point features, rather than one integer address. It is not `[batch, 2]` class scores. `eval()` disables dropout's training behavior; `no_grad()` separately avoids constructing derivative graphs. Both matter for a repeatable inference comparison.

## Why the mask must also reach your own pooling code

Suppose we want one vector per message. **Mean pooling** averages its position vectors. If a two-token message has vectors `[2, 4]` and `[4, 6]`, its mean is `[3, 5]`. Appending padding should not divide that sum by a larger number. BERT can produce nonzero vectors at padded output positions: its input attention mask prevents reading padded keys, but does not turn every padded output vector into zero. A downstream mean must mask them itself.

For hidden tensor `H` and mask `M`, sum `H[b,t,:] * M[b,t]` over positions `t`, then divide by the sum of the mask for that row. `unsqueeze(-1)` gives `[batch, positions, 1]`, so broadcasting applies the mask to all 128 features. Here the mean deliberately includes special tokens; a different pooling policy is a different feature definition. This convenient representation is not a sentence-similarity model or BERT's built-in classification pooler.

```python
if ready:
    weights = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1)
    assert pooled.shape == (4, 128)
    # Same text, extra right padding: real positions and masked mean should agree.
    fixed = tokenizer(messages, padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
    with torch.no_grad():
        fixed_hidden = encoder(**fixed).last_hidden_state
    fixed_weights = fixed["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    fixed_mean = (fixed_hidden * fixed_weights).sum(1) / fixed_weights.sum(1)
    torch.testing.assert_close(pooled, fixed_mean, rtol=1e-4, atol=1e-5)
    print("Dynamic/fixed padded slots:", int((encoded["attention_mask"] == 0).sum()),
          int((fixed["attention_mask"] == 0).sum()))
    print("Masked mean parity passed; unmasked means equal:",
          torch.allclose(hidden.mean(1), fixed_hidden.mean(1)))
```

The equality check concerns padding, not truncation. If a shorter limit removes real tokens, both contextual features and their mean can change. Choosing 64 because the batch fits is an engineering budget; it needs a data measurement too.

```python
if ready:
    train_texts = [row.text for row in splits.train]
    lengths = [len(ids) for ids in tokenizer(train_texts, truncation=False)["input_ids"]]
    length_report = {str(limit): sum(n > limit for n in lengths) for limit in (32, 64)}
    print("Training messages exceeding each total-token limit:", length_report)
    print("Longest training message in tokens:", max(lengths))
```

**Recorded CPU run (2026-10-30):** the first four training messages produced `[4,58]` IDs and `[4,58,128]` contextual features. Dynamic padding used 92 padded slots; fixed width 64 used 116. Masked means agreed within the stated tolerance; unmasked means did not. Of 800 training messages, 223 exceeded 32 total tokens and 11 exceeded 64; the longest had 221. Thus 32 would affect 27.9% of training messages versus 1.4% at 64. These are observations for the frozen split and tokenizer, not a quality benchmark.

A count of messages affected is more useful than silently truncating, but it cannot prove the removed suffixes were unimportant. Chapter 9 holds the 64-token budget fixed for its comparison. The exercise asks you to measure the representation effect at 32; it does not grant permission to tune on test messages.

## Persist the input contract

`save_pretrained` saves the tokenizer's actual vocabulary and settings. The length and pooling choices belong to our application, so we record them separately with the verified base metadata and split identity. A new directory per run prevents overwriting an earlier handoff. The base model remains in the verified cache; Chapter 9 will save adapted weights as well.

```python
if ready:
    import json
    import tempfile
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    input_bundle = Path(tempfile.mkdtemp(prefix="sms-inputs-v1-", dir=ARTIFACTS))
    tokenizer.save_pretrained(input_bundle)
    contract = {
        "task_id": "sms-routing.v1", "label_map": {"ham": 0, "spam": 1},
        "max_length": 64, "padding": "longest", "truncation": True,
        "pooling_demo": "masked mean including special tokens",
        "base": base_metadata, "split_sha256": splits.metadata["split_sha256"],
        "train_length_report": length_report,
    }
    (input_bundle / "input-contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8")
    restored_tokenizer = AutoTokenizer.from_pretrained(
        input_bundle, local_files_only=True, trust_remote_code=False)
    reencoded = restored_tokenizer(messages, padding=True, truncation=True,
                                   max_length=64, return_tensors="pt")
    assert set(reencoded) == set(encoded)
    for key in encoded:
        assert torch.equal(encoded[key], reencoded[key])
    print("Tokenizer round trip passed:", input_bundle.name)
```

## SOLO - measure a different length policy

Implement `inspect_features(tokenizer, encoder, texts, *, max_length=64)`. For a nonempty list of nonblank strings and a total-token limit from 4 through 512, return a dictionary with `features` (a CPU `[N,128]` masked mean), `lengths` (untruncated token counts), and `truncated` (how many counts exceed the limit). Use dynamic padding, inference in evaluation mode and the supplied **real** tokenizer/encoder. Do not change texts or weights. It is acceptable to leave the encoder in evaluation mode. The caller supplies a CPU BERT model of this chapter's width and valid texts/limit; extra validation is not required.

Build on the worked tokenization, pooling and length cells. Python `sum(condition for ...)` counts true values; `.tolist()` converts a tensor into ordinary values if you need to inspect it. The important independent step is connecting those operations in one function, then evaluating a changed limit.

```python exercise
def inspect_features(tokenizer, encoder, texts, *, max_length=64):
    raise NotImplementedError("Tokenize, infer, mask-pool, and report truncation")

chosen_texts = [row.text for row in splits.train[:32]]
short = inspect_features(tokenizer, encoder, chosen_texts, max_length=32)
long = inspect_features(tokenizer, encoder, chosen_texts, max_length=64)
assert short["features"].shape == long["features"].shape == (32, 128)
assert short["truncated"] >= long["truncated"]
print("Affected messages:", short["truncated"], long["truncated"])
print("Mean absolute feature change:",
      (short["features"] - long["features"]).abs().mean().item())
```

Explain why an unchanged shape does not mean unchanged information. Find one training ID whose sequence is longer than 32 and compare its two vectors. If the first 32 contain none, choose the first such ID in the complete training partition. State whether you would keep 64 under our CPU budget and what validation experiment would be needed before shortening it. Do not infer classification improvement from a vector difference alone.

Separate answers and reasoning: `solutions/draft/ch08.py`; tests: `tests/draft/test_ch08.py`. The tests retain the small padding fixture and add real-model padding invariance and tokenizer reload checks.

**Completion gate:** execute the real encoder on public training records; explain IDs, contextual vectors, shapes and masks; demonstrate padding invariance and tokenizer restoration; implement the length comparison and justify its limits. A missing-asset SKIP leaves this gate incomplete. What we carry forward is the frozen data identity, a real BERT input contract and a reusable tokenizer—not yet a trained SMS classifier.
