"""Check the complete learner notebook against its print source and contracts."""
import ast
import importlib.util
import json
from pathlib import Path

import nbformat

from tools.build_ch02_notebook import MANUSCRIPT, NOTEBOOK, build_notebook


def test_complete_notebook_matches_manuscript_and_schema():
    actual = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert actual == build_notebook(MANUSCRIPT.read_text(encoding="utf-8"))
    nbformat.validate(nbformat.from_dict(actual))


def test_exercise_cells_are_separate_and_unexecuted():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    exercises = [c for c in notebook.cells if "exercise" in c.metadata.get("tags", [])]
    assert len(exercises) == 3  # three progressive sample-to-batch exercises
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source)
        assert cell.execution_count is None
        assert cell.outputs == []
        if "raise NotImplementedError" in cell.source:
            assert "exercise" in cell.metadata.get("tags", [])
    worked = "\n".join(c.source for c in notebook.cells if c.cell_type == "code" and c not in exercises)
    assert "ch02_functions" not in worked
    assert "solutions/" not in worked


def test_separate_answers_satisfy_notebook_exercise_examples():
    """Run the actual printed learner assertions using the separate answers."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ch02_answers", root / "solutions/book-01/ch02_functions.py")
    answers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answers)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code" or "exercise" not in cell.metadata.get("tags", []):
            continue
        tree = ast.parse(cell.source)
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {node.name: getattr(answers, node.name) for node in definitions}
        checks = ast.Module(body=[node for node in tree.body if not isinstance(node, ast.FunctionDef)], type_ignores=[])
        exec(compile(checks, str(NOTEBOOK), "exec"), namespace)
