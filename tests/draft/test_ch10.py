import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("draft10", Path(__file__).resolve().parents[2] / "solutions/draft/ch10.py")
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


def test_quarantine_preserves_rows_and_line_numbers_without_raw_payloads():
    lines = ['{"id":"a","text":"Здравей"}', ' ', '{bad', '[]', '{"id":"","text":"x"}',
             '{"id":"b","text":17}', '{"id":"a","text":"duplicate"}']
    good, rejected = answer.quarantine_jsonl(iter(lines))
    assert good == [{"id": "a", "text": "Здравей"}]
    assert [r["line"] for r in rejected] == [3, 4, 5, 6, 7]
    assert all(set(r) == {"line", "reason"} for r in rejected)
    assert rejected[-1]["reason"] == "duplicate id"


def test_empty_and_invalid_first_occurrence():
    assert answer.quarantine_jsonl([]) == ([], [])
    good, bad = answer.quarantine_jsonl(['{"id":"a"}', '{"id":"a","text":"ok"}'])
    assert good == [{"id": "a", "text": "ok"}]
    assert bad == [{"line": 1, "reason": "invalid text"}]
