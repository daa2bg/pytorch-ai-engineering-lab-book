"""Reference answer: shifted character pairs, not a pretrained model."""
import torch


def next_token_pairs(text: str, vocabulary: dict[str, int]):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) < 2:
        raise ValueError("at least two characters are required")
    ids = torch.tensor([vocabulary[character] for character in text], dtype=torch.long)
    return ids[:-1], ids[1:]
