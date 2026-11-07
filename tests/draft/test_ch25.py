import pytest
from solutions.draft.ch25 import task_fingerprint


def test_fingerprint_is_stable_for_dictionary_order():
    assert task_fingerprint({"id": "04", "topic": "tensors"}, {"style": "short"}) == task_fingerprint({"topic": "tensors", "id": "04"}, {"style": "short"})


def test_fingerprint_changes_when_requirements_change():
    original = task_fingerprint({"id": "04"}, {"style": "short"})
    assert original != task_fingerprint({"id": "04"}, {"style": "long"})
    assert original != task_fingerprint({"id": "05"}, {"style": "short"})


def test_fingerprint_rejects_wrong_containers():
    with pytest.raises(TypeError):
        task_fingerprint([], {})
