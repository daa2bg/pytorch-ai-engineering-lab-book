import pytest

from solutions.draft.ch03 import PairDataset


def test_pair_dataset_alignment_and_snapshot():
    rows, labels = [[2.0, 5.0], [1.0, 3.0]], [0, 1]
    dataset = PairDataset(rows, labels)
    rows[0][0] = 100
    rows.append([9, 9])
    labels[0] = 7
    assert len(dataset) == 2
    assert dataset[0] == ((2.0, 5.0), 0)
    assert dataset[-1] == ((1.0, 3.0), 1)


def test_pair_dataset_empty_and_bad_index():
    dataset = PairDataset([], [])
    assert len(dataset) == 0
    with pytest.raises(IndexError):
        dataset[0]


def test_pair_dataset_rejects_misalignment():
    with pytest.raises(ValueError):
        PairDataset([[1.0]], [])


def test_loader_preserves_snapshot_alignment_and_tail_across_epochs():
    import torch
    from torch.utils.data import DataLoader
    from solutions.draft.ch03 import collate_pairs

    rows, labels = [[2.0, 5.0], [1.0, 3.0], [4.0, 2.0]], [0, 1, 0]
    dataset = PairDataset(rows, labels)
    rows[0][0], labels[0] = 999.0, 1
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_pairs,
                        num_workers=0, shuffle=False)
    for _ in range(2):
        batches = list(loader)
        assert [len(y) for x, y in batches] == [2, 1]
        assert torch.cat([x for x, y in batches]).tolist() == [
            [2.0, 5.0], [1.0, 3.0], [4.0, 2.0],
        ]
        assert torch.cat([y for x, y in batches]).tolist() == [0, 1, 0]
        assert batches[0][0].dtype == torch.float32
        assert batches[0][1].dtype == torch.long
    with pytest.raises(IndexError):
        dataset[3]
