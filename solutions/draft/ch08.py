"""Study 08 reference answer: an offline token-batch adapter, not a tokenizer."""


def pad_batch(sequences, pad_id=0):
    width = max((len(sequence) for sequence in sequences), default=0)
    input_ids = []
    attention_mask = []
    for sequence in sequences:
        missing = width - len(sequence)
        input_ids.append(list(sequence) + [pad_id] * missing)
        attention_mask.append([1] * len(sequence) + [0] * missing)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def inspect_features(tokenizer, encoder, texts, *, max_length=64):
    """Run BERT and compare a bounded input policy without fitting on held-out text.

    Masking is required again after the encoder: padded output positions need not
    be zero. Include special tokens, matching the chapter's explicit convention.
    Limits change information, so equal feature shapes do not establish parity.
    The caller supplies valid nonempty texts, CPU encoder, and a 4..512 limit.
    """
    import torch
    lengths = [len(row) for row in tokenizer(texts, truncation=False)["input_ids"]]
    batch = tokenizer(texts, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    encoder.eval()
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state
        weights = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        features = (hidden * weights).sum(1) / weights.sum(1)
    return {"features": features.cpu(), "lengths": lengths,
            "truncated": sum(n > max_length for n in lengths)}
