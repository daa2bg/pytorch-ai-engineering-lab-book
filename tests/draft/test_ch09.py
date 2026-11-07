import os
import pytest
import torch
from solutions.draft.ch09 import adapt_last_layer


def test_empty_loader_rejects():
    from transformers import BertConfig, BertForSequenceClassification
    # A random tiny fixture checks only the empty-loop contract.
    model = BertForSequenceClassification(BertConfig(
        vocab_size=16, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=16, num_labels=2))
    with pytest.raises(ValueError, match="no samples"):
        adapt_last_layer(model, [], epochs=1)


def test_real_three_epoch_adaptation_and_reload(tmp_path, monkeypatch):
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader
    from ai_lab.projects.assets import resolve_asset, asset_metadata
    from ai_lab.projects.sms import (load_sms_records, make_sms_splits,
                                    load_sms_splits, classification_metrics)
    try:
        path = resolve_asset("bert", allow_downloads=False)
        asset_metadata(path)
        saved = Path(os.environ.get("AI_LAB_ARTIFACT_DIR", "artifacts/projects")) / "sms-splits-v1"
        splits = (load_sms_splits(saved) if saved.exists() else
                  make_sms_splits(load_sms_records(allow_downloads=False)))
    except FileNotFoundError:
        if os.environ.get("AI_LAB_RUN_PRETRAINED") == "1":
            raise
        pytest.skip("Required real BERT/SMS unavailable; adaptation outcome incomplete")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=2, id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1}, local_files_only=True, trust_remote_code=False)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    counts, updates = [], []
    def collate(rows):
        batch = tokenizer([r.text for r in rows], padding=True, truncation=True,
                          max_length=64, return_tensors="pt")
        batch["labels"] = torch.tensor([r.label for r in rows])
        counts.append(len(rows))
        return batch
    loader = DataLoader(splits.train, batch_size=16, shuffle=True,
                        generator=torch.Generator().manual_seed(42), collate_fn=collate)
    original_step = torch.optim.AdamW.step
    def counted_step(self, *args, **kwargs):
        updates.append(1)
        return original_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    losses = adapt_last_layer(model, loader)
    assert len(losses) == 3 and all(torch.isfinite(torch.tensor(losses)))
    assert len(updates) == 150 and sum(counts) == 2400
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert any(n.startswith("classifier.") for n in changed)
    assert any(n.startswith("bert.encoder.layer.1.") for n in changed)
    assert all(n.startswith(("classifier.", "bert.encoder.layer.1.")) for n in changed)
    assert not any(p.grad is not None for p in model.bert.embeddings.parameters())
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(splits.validation, batch_size=16, collate_fn=collate):
            predictions.extend(model(**batch).logits.argmax(1).tolist())
    metrics = classification_metrics([r.label for r in splits.validation], predictions)
    print("RATE_VARIANT", {"losses": losses, "validation": metrics,
                           "changed_tensors": len(changed), "updates": len(updates)})
    # Independently restored tokenizer must create inputs for the restored weights.
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    restored_tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True,
                                                      trust_remote_code=False)
    restored = AutoModelForSequenceClassification.from_pretrained(
        tmp_path, local_files_only=True, trust_remote_code=False).eval()
    text = [r.text for r in splits.validation[:4]]
    original_inputs = tokenizer(text, padding=True, truncation=True, max_length=64, return_tensors="pt")
    restored_inputs = restored_tokenizer(text, padding=True, truncation=True,
                                        max_length=64, return_tensors="pt")
    for key in original_inputs:
        assert torch.equal(original_inputs[key], restored_inputs[key])
    with torch.no_grad():
        torch.testing.assert_close(model(**original_inputs).logits,
                                   restored(**restored_inputs).logits, rtol=0, atol=0)
