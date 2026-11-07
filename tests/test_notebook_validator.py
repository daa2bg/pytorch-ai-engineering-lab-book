import json
from pathlib import Path
from tools.validate_notebooks import code_cells, exercise, validate

def test_exercise_detection():
    assert exercise({"cell_type":"code","metadata":{"tags":["exercise"]},"source":["x=1"]})
    assert exercise({"cell_type":"code","metadata":{},"source":["raise NotImplementedError"]})
    assert not exercise({"cell_type":"code","metadata":{},"source":["assert 1 == 1"]})

def test_reference_cells_execute_and_exercise_skipped(tmp_path):
    notebook = {"nbformat":4,"cells":[
        {"cell_type":"code","metadata":{},"source":["x=2"]},
        {"cell_type":"code","metadata":{"tags":["exercise"]},"source":["raise RuntimeError('should skip')"]},
        {"cell_type":"code","metadata":{},"source":["assert x == 2"]}
    ]}
    path=tmp_path/"sample.ipynb"
    path.write_text(json.dumps(notebook))
    assert len(code_cells(path)) == 2
    ok, message = validate(path, 10)
    assert ok, message

def test_reference_failure_is_detected(tmp_path):
    path=tmp_path/"bad.ipynb"
    path.write_text(json.dumps({"nbformat":4,"cells":[{"cell_type":"code","metadata":{},"source":["assert False"]}]}))
    ok, _ = validate(path, 10)
    assert not ok
