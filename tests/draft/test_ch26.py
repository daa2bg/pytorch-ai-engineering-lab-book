import copy
import pytest
from solutions.draft.ch26 import validate_records


def test_records_are_canonical_and_independent():
    source = [{"id": " a ", "text": " invoice ", "label": "invoice", "extra": 1}]
    before = copy.deepcopy(source)
    result = validate_records(iter(source))
    assert result == [{"id": "a", "text": "invoice", "label": "invoice"}]
    assert source == before and result[0] is not source[0]
    assert validate_records([]) == []


@pytest.mark.parametrize("records", [[{"id":"x", "text":"x"}], [{"id":"", "text":"x", "label":"claim"}], [{"id":"x", "text":" ", "label":"claim"}], [{"id":"x", "text":"x", "label":"other"}]])
def test_reject_invalid_values(records):
    with pytest.raises(ValueError):
        validate_records(records)


def test_duplicate_ids_are_detected_after_normalization():
    with pytest.raises(ValueError, match="duplicate"):
        validate_records([{"id": key, "text": "x", "label": "claim"} for key in ["x", " x "]])


def test_wrong_type_and_nonstring():
    with pytest.raises(TypeError):
        validate_records([17])
    with pytest.raises(TypeError):
        validate_records([{"id": "x", "text": 17, "label": "claim"}])
