# Study 21 — Train a tiny next-character model

**First draft; optional architecture study.** You have already trained a classifier and used pretrained models. Now make the language-model objective visible in a system small enough to inspect. This experiment uses CPU PyTorch and local text; it downloads nothing.

A language model assigns probabilities to possible continuations. In next-token training, the target at each position is the token immediately after the input context. Here a token is one character and the context is just the current character. The result is a **bigram model**, not a transformer or a useful assistant. This deliberately limited model separates the training objective from sophisticated architecture.

## Construct the prediction task

Consider `aba`: its supervised examples are `a → b` and `b → a`. We do not give the model `a → a` at the same position, because that would teach copying instead of continuation. Python slicing expresses the shift directly. The vocabulary maps characters to integer IDs; those IDs are identifiers, not numerical measurements.

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(7)
torch.set_num_threads(1)
text = "ab ab ab ab ab ab "
characters = sorted(set(text))
stoi = {character: index for index, character in enumerate(characters)}
ids = torch.tensor([stoi[character] for character in text], dtype=torch.long)
x, y = ids[:-1], ids[1:]
assert len(x) == len(y) == len(text) - 1
assert text[1] == characters[y[0].item()]
print("Vocabulary:", stoi, "Training pairs:", len(x))
```

Each output must contain one score per vocabulary item. `nn.Embedding(vocabulary_size, vocabulary_size)` is a learned lookup table: looking up the current character returns its scores for the next character. An embedding usually represents tokens as shorter vectors, but nothing prevents its output width from equaling the vocabulary size here. Cross-entropy compares raw scores with integer target IDs; applying softmax before that loss would be redundant and numerically undesirable.

```python
model = nn.Embedding(len(characters), len(characters))
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
with torch.no_grad():
    initial_loss = F.cross_entropy(model(x), y).item()
for step in range(40):
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()
with torch.no_grad():
    final_loss = F.cross_entropy(model(x), y).item()
assert final_loss < initial_loss
print("Observed training loss:", initial_loss, "->", final_loss)
```

The assertion checks that this bounded optimization learned something about its training corpus. It does not demonstrate generalization. Repeated training on one tiny string can memorize transitions; there is no independent test corpus here. The exact printed losses are observations from your environment, not a performance benchmark.

## Generate from the learned distribution

Generation repeatedly predicts, selects a continuation, appends it, and uses that character as the next input. **Temperature** rescales scores before softmax. Lower positive values concentrate probability on high-scoring characters; higher values spread it out. Sampling uses a separately seeded generator so you can repeat the experiment without resetting model initialization.

```python
generator = torch.Generator().manual_seed(11)
current = torch.tensor([stoi["a"]])
output = ["a"]
with torch.no_grad():
    for _ in range(12):
        probabilities = torch.softmax(model(current) / 0.7, dim=-1)
        current = torch.multinomial(probabilities, 1, generator=generator).reshape(1)
        output.append(characters[current.item()])
assert len(output) == 13
assert set(output) <= set(characters)
print("Generated:", "".join(output))
```

Even perfect transition estimates cannot express long-range dependencies with this architecture. A model that only sees the previous character cannot distinguish two histories ending in the same character. Longer context requires a different representation and architecture, not simply more epochs.

## SOLO exercise — Make the shift a reusable contract

Implement `next_token_pairs(text, vocabulary)`, returning two `torch.long` tensors containing all adjacent input/target pairs. Reject non-string text with `TypeError`, fewer than two characters with `ValueError`, and let an unknown character raise `KeyError`. Do not add an artificial end token or silently discard unknown characters.

```python exercise
def next_token_pairs(text, vocabulary):
    raise NotImplementedError("Build shifted input and target tensors")

example_x, example_y = next_token_pairs("aba", {"a": 0, "b": 1})
assert example_x.tolist() == [0, 1]
assert example_y.tolist() == [1, 0]
```

**Optional experiment:** hold out a different string, measure its loss, then compare with training loss. Change the corpus while preserving the vocabulary. Explain why a character absent from training needs an explicit unknown-token or vocabulary policy.

**Completion gate:** produce shifted pairs, reduce the observed training loss, generate a bounded continuation, and explain why this is a character bigram experiment rather than a tiny GPT. Reference answer: `solutions/draft/ch21.py`; tests: `tests/draft/test_ch21.py`.

Primary references: [PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html), [cross-entropy](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.cross_entropy.html), and [Hugging Face generation strategies](https://huggingface.co/docs/transformers/generation_strategies). Context windows and causal attention are developed in Study 20; the small objective here also applies when an architecture uses a longer context.
