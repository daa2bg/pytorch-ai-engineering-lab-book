import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft12", Path(__file__).resolve().parents[2] / "solutions/draft/ch12.py")
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


def test_fingerprint_canonicalizes_keys_but_not_content_or_order():
    first = [{"id": "a", "text": "claim"}, {"id": "b", "text": "invoice"}]
    reordered_keys = [{"text": "claim", "id": "a"}, {"text": "invoice", "id": "b"}]
    fp = answer.dataset_fingerprint(first, {"lowercase": True})
    assert len(fp) == 64
    assert fp == answer.dataset_fingerprint(reordered_keys, {"lowercase": True})
    assert fp != answer.dataset_fingerprint(first[::-1], {"lowercase": True})
    assert fp != answer.dataset_fingerprint(first, {"lowercase": False})
    assert first[0]["text"] == "claim"


def test_nonstandard_numbers_rejected():
    with pytest.raises(ValueError):
        answer.dataset_fingerprint([float("nan")], {})
