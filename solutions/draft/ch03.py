"""Study 03 reference answer: an owned snapshot of aligned samples."""


class PairDataset:
    """Snapshot numeric rows and scalar labels; integer indexing follows tuples.

    A shallow outer snapshot would still share mutable feature rows. Copy each
    row into a tuple while preserving the pairing and original order.
    """

    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise ValueError("features and labels must have equal lengths")
        self.features = tuple(tuple(row) for row in features)
        self.labels = tuple(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def collate_pairs(samples):
    """Worked batch callback, included for the independent loader tests."""
    import torch

    rows = [features for features, label in samples]
    labels = [label for features, label in samples]
    return (torch.tensor(rows, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))
