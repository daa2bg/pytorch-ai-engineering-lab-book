"""Observable values, ownership, and failure contracts for Chapter 2 labs."""
import copy
import importlib.util
from pathlib import Path

import pytest

SOLUTION = Path(__file__).resolve().parents[1] / "solutions/book-01/ch02_functions.py"
spec = importlib.util.spec_from_file_location("ch02_solutions", SOLUTION)
answers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answers)


@pytest.mark.parametrize("text,expected", [
    ("Claim for repairs", [3.0, 5.0]),
    ("  Invoice\t due\n", [2.0, 5.0]),
    ("", [0.0, 0.0]),
    (" \t\n ", [0.0, 0.0]),
    ("Hi, WORLD!", [2.0, 4.5]),
    ("ЗДРАВЕЙ свят", [2.0, 5.5]),
    ("café ☕", [2.0, 2.5]),
    ("İ", [1.0, 2.0]),  # Lowercase is applied before measuring code points.
])
def test_text_feature_values(text, expected):
    result = answers.text_features(text)
    assert result == expected
    assert all(isinstance(value, float) for value in result)


@pytest.mark.parametrize("bad", [None, 17, 2.5, b"claim", ["claim"], {"text": "claim"}])
def test_text_features_rejects_nonstring(bad):
    with pytest.raises(TypeError, match="text"):
        answers.text_features(bad)


def test_feature_results_are_independent_even_when_empty():
    first = answers.text_features("")
    second = answers.text_features("")
    assert first is not second
    first[0] = 99.0
    assert second == [0.0, 0.0]


def record():
    return {
        "id": "doc-1", "text": " Invoice due ", "label": "invoice",
        "metadata": {"tags": ["urgent"]},
    }


def test_prepared_sample_schema_values_and_preservation():
    source = record()
    before = copy.deepcopy(source)
    mapping = {"claim": 9, "invoice": 2}
    result = answers.prepare_sample(source, mapping)
    assert result == {"id": "doc-1", "features": [2.0, 5.0], "label_id": 2}
    assert result is not source
    assert source == before
    assert mapping == {"claim": 9, "invoice": 2}
    result["features"].append(99.0)
    result["id"] = "changed"
    assert source == before
    assert answers.prepare_sample(source, mapping)["features"] == [2.0, 5.0]


@pytest.mark.parametrize("mapping,expected", [({"invoice": 0}, 0), ({"invoice": 42}, 42)])
def test_prepared_sample_uses_supplied_mapping(mapping, expected):
    assert answers.prepare_sample(record(), mapping)["label_id"] == expected


@pytest.mark.parametrize("missing", ["id", "text", "label"])
def test_prepared_sample_missing_fields_raise_keyerror(missing):
    source = record()
    del source[missing]
    with pytest.raises(KeyError) as error:
        answers.prepare_sample(source, {"invoice": 2})
    assert error.value.args == (missing,)


def test_prepared_sample_does_not_silently_normalize_unknown_labels():
    source = record()
    source["label"] = " Invoice "
    with pytest.raises(KeyError) as error:
        answers.prepare_sample(source, {"invoice": 2})
    assert error.value.args == (" Invoice ",)


@pytest.mark.parametrize("bad", [None, 17, ["Invoice"]])
def test_prepared_sample_propagates_text_errors_without_mutation(bad):
    source = record()
    source["text"] = bad
    before = copy.deepcopy(source)
    with pytest.raises(TypeError):
        answers.prepare_sample(source, {"invoice": 2})
    assert source == before


def records():
    return [
        {"id": "z", "text": "Invoice due", "label": "invoice"},
        {"id": "a", "text": "Claim for repairs", "label": "claim"},
        {"id": "m", "text": "", "label": "invoice"},
    ]


def test_batch_preserves_order_alignment_and_inputs():
    source = records()
    before = copy.deepcopy(source)
    mapping = {"claim": 9, "invoice": 2}
    batch = answers.make_batch(source, mapping)
    assert isinstance(batch, tuple)
    assert batch == ([[2.0, 5.0], [3.0, 5.0], [0.0, 0.0]], [2, 9, 2])
    assert source == before
    assert mapping == {"claim": 9, "invoice": 2}


@pytest.mark.parametrize("source", [[], iter(())])
def test_empty_batch(source):
    assert answers.make_batch(source, {}) == ([], [])


def test_batch_accepts_and_exhausts_generator():
    source = (item for item in records())
    assert answers.make_batch(source, {"invoice": 2, "claim": 9})[1] == [2, 9, 2]
    assert list(source) == []


def test_batch_traverses_source_once():
    class OnePass:
        def __init__(self):
            self.visits = 0

        def __iter__(self):
            self.visits += 1
            if self.visits > 1:
                raise AssertionError("second traversal")
            yield from records()

    source = OnePass()
    assert answers.make_batch(source, {"invoice": 2, "claim": 9})[1] == [2, 9, 2]
    assert source.visits == 1


def test_batch_rows_and_calls_do_not_share_mutable_lists():
    source_record = record()
    source = [source_record, source_record]
    before = copy.deepcopy(source)
    features, labels = answers.make_batch(source, {"invoice": 2})
    assert features[0] is not features[1]
    features[0][0] = 99.0
    labels[0] = -1
    assert features[1] == [2.0, 5.0]
    assert source == before
    assert answers.make_batch(source, {"invoice": 2}) == (
        [[2.0, 5.0], [2.0, 5.0]], [2, 2]
    )


@pytest.mark.parametrize("bad,expected", [
    ({"id": "bad", "label": "invoice"}, KeyError),
    ({"id": "bad", "text": "text", "label": "unknown"}, KeyError),
    ({"id": "bad", "text": None, "label": "invoice"}, TypeError),
])
def test_batch_propagates_preparation_error_and_stops_at_bad_record(bad, expected):
    good = record()
    source_records = [good, bad, good]
    before = copy.deepcopy(source_records)
    source = iter(source_records)
    with pytest.raises(expected):
        answers.make_batch(source, {"invoice": 2})
    assert next(source) is good
    assert source_records == before


def test_batch_propagates_source_error():
    def interrupted():
        yield record()
        raise RuntimeError("source interrupted")

    with pytest.raises(RuntimeError, match="source interrupted"):
        answers.make_batch(interrupted(), {"invoice": 2})
