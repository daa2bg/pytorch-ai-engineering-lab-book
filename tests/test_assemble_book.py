"""Assembled drafts must be complete and must not expose execution outputs."""
from tools.assemble_book import assemble, notebook_markdown


def test_reference_notebook_preserves_source_without_saved_output():
    notebook = {"cells": [
        {"cell_type": "markdown", "source": ["Explain the example."]},
        {"cell_type": "code", "metadata": {"tags": ["exercise"]},
         "source": ["raise NotImplementedError"],
         "outputs": [{"text": "saved output must not be copied"}]},
    ]}
    result = notebook_markdown(notebook)
    assert "Explain the example." in result
    assert "```python exercise\nraise NotImplementedError\n```" in result
    assert "saved output must not be copied" not in result


def test_missing_managed_chapter_is_reported_even_when_notebook_exists(tmp_path):
    (tmp_path / "study.ipynb").write_text('{"cells": []}')
    manifest = {"chapters": [{"id": "03", "title": "Study", "tier": "core",
        "managed": True, "paths": {"manuscript": "missing.md", "notebook": "study.ipynb"}}]}
    _, missing = assemble(tmp_path, manifest)
    assert missing == ["03"]


def test_complete_manuscript_is_included_in_order(tmp_path):
    (tmp_path / "a.md").write_text("First study contents.")
    (tmp_path / "b.md").write_text("Second study contents.")
    manifest = {"chapters": [
        {"id": "03", "title": "A", "tier": "core", "paths": {"manuscript": "a.md", "notebook": "a.ipynb"}},
        {"id": "04", "title": "B", "tier": "core", "paths": {"manuscript": "b.md", "notebook": "b.ipynb"}},
    ]}
    contents, missing = assemble(tmp_path, manifest)
    assert missing == []
    assert contents.index("First study contents.") < contents.index("Second study contents.")
    assert "not a release candidate" in contents
