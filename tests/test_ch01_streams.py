"""Exercise contracts and the protected chapter's print/notebook packaging."""
import ast
import importlib.util
import io
import json
from pathlib import Path

import nbformat
import pytest

from tools.book_factory import build_notebook

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ch01", ROOT / "solutions/book-01/ch01_streams.py")
answers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answers)


def test_one_shot_batching_preserves_tail_and_consumes_lazily():
    visited = []
    def source():
        for value in range(5):
            visited.append(value)
            yield value
    result = answers.my_batches(source(), 2)
    assert visited == []
    assert next(result) == [0, 1]
    assert visited == [0, 1]
    assert list(result) == [[2, 3], [4]]


def test_fresh_epochs_skip_empty_text_without_mutating_records():
    records = [{"text": " first "}, {"text": " "}, {"text": "second"}]
    assert answers.count_epochs(records, 3) == [2, 2, 2]
    assert answers.count_epochs(records, 0) == []
    assert records[0]["text"] == " first "
    with pytest.raises(ValueError):
        answers.count_epochs(records, -1)


def test_jsonl_stream_closes_after_partial_consumption(monkeypatch):
    stream = io.StringIO('\n{"id":"a","text":"hello","label":"general","extra":1}\n')
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: stream)
    rows = answers.read_documents("unused")
    assert next(rows)["extra"] == 1
    assert not stream.closed
    rows.close()
    assert stream.closed


@pytest.mark.parametrize("bad", ['not json', '[]', '{"id":"missing"}'])
def test_jsonl_reports_physical_line_and_closes_on_error(monkeypatch, bad):
    stream = io.StringIO('\n' + bad + '\n')
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: stream)
    with pytest.raises(ValueError, match="line 2"):
        list(answers.read_documents("unused"))
    assert stream.closed


def test_unicode_and_missing_file_policy(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text('{"id":"1","text":"Здравей","label":"general"}\n', encoding="utf-8")
    assert list(answers.read_documents(source))[0]["text"] == "Здравей"
    with pytest.raises(FileNotFoundError):
        list(answers.read_documents(tmp_path / "missing.jsonl"))
    source.write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        list(answers.read_documents(source))


def test_print_notebook_parity_and_separate_unfinished_exercises():
    relative = "book/01-python-for-ai/01-iterators-and-batching.md"
    expected = build_notebook((ROOT / relative).read_text(encoding="utf-8"), relative)
    actual = nbformat.read(ROOT / relative.replace(".md", ".ipynb"), as_version=4)
    nbformat.validate(actual)
    assert json.loads((ROOT / relative.replace(".md", ".ipynb")).read_text(encoding="utf-8")) == expected
    exercises = [c for c in actual.cells if "exercise" in c.metadata.get("tags", [])]
    assert len(exercises) == 3
    assert all("raise NotImplementedError" in c.source for c in exercises)
    worked = "\n".join(c.source for c in actual.cells if c.cell_type == "code" and c not in exercises)
    assert "ch01_streams" not in worked
    assert "epoch_counts = []" not in worked


def test_printed_exercise_checks_with_separate_answers():
    notebook = nbformat.read(ROOT / "book/01-python-for-ai/01-iterators-and-batching.ipynb", as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        if "exercise" not in cell.metadata.get("tags", []):
            exec(compile(cell.source, "chapter01", "exec"), namespace)
        else:
            tree = ast.parse(cell.source)
            for definition in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
                namespace[definition.name] = getattr(answers, definition.name)
            checks = ast.Module(body=[n for n in tree.body if not isinstance(n, ast.FunctionDef)], type_ignores=[])
            exec(compile(checks, "chapter01-exercise-checks", "exec"), namespace)
