# Study 19 — Images and Sequences Change the Shape Contract

**First draft. Optional architecture study, about 45 minutes.** You can train and fine-tune a pretrained model before completing this study. Return here when image or sequence structure affects the model you choose. The goal is to trace a small CNN and recurrent encoder, then prevent padded sequence positions from corrupting a pooled representation. All code uses small CPU tensors and downloads nothing.

## An image is more than a flattened row

A convolutional neural network (**CNN**) applies learned filters across spatial positions. Sharing a filter lets the same local pattern detector operate in different regions. This gives a useful structural assumption for images, but does not guarantee that a trained model will recognize every shift, rotation, or document layout.

PyTorch's two-dimensional convolution expects a batch with axes `[batch, channels, height, width]`. A grayscale image has one channel; an RGB image normally has three. The network below uses synthetic images to verify shape flow. Randomly initialized filters are not a trained vision system.

```python
import torch
from torch import nn

torch.manual_seed(19)
images = torch.randn(2, 1, 8, 8)
convolution = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
activation_maps = torch.relu(convolution(images))
assert tuple(activation_maps.shape) == (2, 4, 6, 6)
pooled_images = activation_maps.mean(dim=(2, 3))
classifier = nn.Linear(4, 3)
scores = classifier(pooled_images)
assert tuple(scores.shape) == (2, 3)
print("Images -> maps -> pooled -> class scores:",
      images.shape, activation_maps.shape, pooled_images.shape, scores.shape)
```

With no padding and stride one, a width-three filter can occupy six positions along an eight-element axis. Four learned filters produce four output channels. Taking the mean over the height and width axes leaves one number per channel. A linear head maps those four numbers to three class scores. Labels would have shape `[2]` for the class-index loss contract used earlier.

Do not confuse this manual pooling with a universal best design. For page images, resizing, aspect ratios, text resolution, augmentation, and a pretrained backbone may matter more than the toy architecture choice. Always evaluate on representative documents rather than synthetic shape fixtures.

## A sequence carries order

A recurrent layer processes positions in sequence and carries a hidden state. A gated recurrent unit (**GRU**) learns how to update that state. With `batch_first=True`, its input layout is `[batch, time, features]`. Here each of three positions has four input features; these could be embeddings or sensor measurements.

```python
sequence = torch.randn(2, 3, 4)
encoder = nn.GRU(input_size=4, hidden_size=5, batch_first=True)
outputs, final_state = encoder(sequence)
assert tuple(outputs.shape) == (2, 3, 5)
assert tuple(final_state.shape) == (1, 2, 5)
print("Per-position representations:", outputs.shape)
```

`outputs` contains a hidden representation at every time position. The leading axis of `final_state` counts recurrent layers and directions, not batch size. A single one-direction layer gives size one there. LSTM layers add a separate cell state; do not substitute their return structure without checking the API contract.

## Padding is storage, not evidence

Real sequences have different lengths. A batch may pad the shorter ones to a common width. Those extra positions should not influence a pooled feature merely because they occupy storage. Even padding with zero changes an unmasked average: dividing by the padded width shrinks a short sequence's representation.

```python
representations = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]]])
valid_length = 2
expected = representations[0, :valid_length].mean(dim=0)
assert torch.equal(expected, torch.tensor([2.0, 4.0]))
assert not torch.allclose(representations.mean(dim=1)[0], expected)
print("The padded value must not contribute:", expected)
```

This checks pooling, not recurrent padding behavior. A recurrent layer can also update its state while consuming padded positions. Packed sequences or a suitable length-aware strategy are separate choices when building an actual recurrent classifier.

### SOLO exercise — length-aware pooling

Write `masked_mean(batch, lengths)` for floating-point input `[B,T,D]` and integer lengths `[B]`. Return `[B,D]`, averaging only positions before each length. Reject incompatible ranks/batch sizes or lengths outside `1..T` with `ValueError`; reject nonfloating input or noninteger lengths with `TypeError`. Do not change the input. The caller supplies finite values and compatible devices; moving lengths to the batch device is permitted. A Boolean mask from `torch.arange(T)` and broadcasting can mark valid positions.

```python exercise
def masked_mean(batch, lengths):
    raise NotImplementedError("Pool only genuine sequence positions")

values = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
assert torch.equal(masked_mean(values, torch.tensor([2])), torch.tensor([[2., 4.]]))
```

**Completion gate:** explain every shape above and prove changing a padded value cannot change the pooled result. **Optional extension:** verify that padded positions receive zero gradient through your pooler, then compare a packed GRU with an unpacked, padded GRU on a tiny fixture.

Reference answers: `solutions/draft/ch19.py`; tests: `tests/draft/test_ch19.py`. Primary sources: [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html), and [packed sequences](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html). Next, study a different way to mix information across sequence positions: attention.
