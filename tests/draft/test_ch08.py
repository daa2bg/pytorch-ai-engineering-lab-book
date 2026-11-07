from solutions.draft.ch08 import pad_batch


def test_padding_values_masks_and_source_independence():
    source = [[2, 3], [4]]
    result = pad_batch(source)
    assert result == {"input_ids": [[2, 3], [4, 0]], "attention_mask": [[1, 1], [1, 0]]}
    result["input_ids"][0][0] = 99
    assert source == [[2, 3], [4]]


def test_padding_uses_lengths_not_value_equality():
    assert pad_batch([[0], [7, 8]], pad_id=0) == {
        "input_ids": [[0, 0], [7, 8]], "attention_mask": [[1, 0], [1, 1]],
    }


def test_padding_empty_batches_and_custom_id():
    assert pad_batch([]) == {"input_ids": [], "attention_mask": []}
    assert pad_batch([[], []]) == {"input_ids": [[], []], "attention_mask": [[], []]}
    assert pad_batch([[], [4]], pad_id=9) == {
        "input_ids": [[9], [4]], "attention_mask": [[0], [1]],
    }


def test_real_encoder_padding_truncation_and_saved_tokenizer(tmp_path):
    """Runs automatically with a verified cache; strict CI cannot skip assets."""
    import os
    import pytest
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from ai_lab.projects.assets import resolve_asset, asset_metadata
    from transformers import AutoModel, AutoTokenizer
    from solutions.draft.ch08 import inspect_features
    try:
        path = resolve_asset("bert", allow_downloads=False)
        asset_metadata(path)
    except FileNotFoundError:
        if os.environ.get("AI_LAB_RUN_PRETRAINED") == "1":
            raise
        pytest.skip("Required real BERT asset unavailable; not pretrained evidence")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                             trust_remote_code=False)
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    texts = ["A message arrived.", "A message " * 45]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    one = inspect_features(tokenizer, model, texts[:1])
    both = inspect_features(tokenizer, model, texts)
    short = inspect_features(tokenizer, model, texts, max_length=32)
    torch.testing.assert_close(one["features"][0], both["features"][0], atol=1e-5, rtol=1e-4)
    assert both["features"].shape == (2, 128)
    assert both["truncated"] == short["truncated"] == 1
    assert short["lengths"] == both["lengths"]
    assert not torch.allclose(short["features"][1], both["features"][1])
    assert not both["features"].requires_grad
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
    assert texts == ["A message arrived.", "A message " * 45]
    tokenizer.save_pretrained(tmp_path)
    restored = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True,
                                            trust_remote_code=False)
    again = inspect_features(restored, model, texts)
    torch.testing.assert_close(again["features"], both["features"], rtol=0, atol=0)
