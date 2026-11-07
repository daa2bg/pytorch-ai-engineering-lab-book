"""Chapter 09 reference answer: actual final-layer adaptation, not name filtering.

A larger encoder learning rate is an experimental choice; it is not guaranteed to
beat the worked recipe. Compare final-epoch validation with identical data/budget.
Never choose the rate by repeatedly scoring the held-out test partition.
"""


def adapt_last_layer(model, train_loader, *, encoder_lr=1e-4, epochs=3):
    """Mutate the supplied BERT, returning sample-weighted pre-update losses.

    Freeze everything first so a previously modified requires_grad policy cannot
    accidentally train embeddings or the pooler. eval() disables base dropout but
    leaves gradient computation through the enabled final layer intact.
    """
    import torch
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    last = list(model.bert.encoder.layer[-1].parameters())
    head = list(model.classifier.parameters())
    for parameter in last + head:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": head, "lr": 1e-3},
        {"params": last, "lr": encoder_lr},
    ], weight_decay=0.01)
    history = []
    for _ in range(epochs):
        model.train()
        model.bert.eval()
        total, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            count = len(batch["labels"])
            total += loss.item() * count
            seen += count
        if not seen:
            raise ValueError("training loader yielded no samples")
        history.append(total / seen)
    return history

# Observed reference run (2026-10-30, same 800/200 saved train/validation IDs):
# encoder_lr=1e-4, 3 epochs: training loss 0.24337, 0.07987, 0.05579;
# validation confusion [[168,7],[1,24]], macro-F1 0.91694, accuracy 0.96.
# Worked final epoch at 5e-5 had macro-F1 0.9051. The rate variant catches more
# validation spam but produces more false positives. This is not a test-set
# selection result and does not establish a universal learning-rate advantage.
