import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft17", Path(__file__).parents[2] / "solutions/draft/ch17.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_restart_reuses_persisted_result(tmp_path):
    path = tmp_path / "jobs.sqlite"
    assert m.run_once(path, "job-1", lambda: {"label": "claim"}) == {"label": "claim"}
    assert m.run_once(path, "job-1", lambda: pytest.fail("duplicate")) == {"label": "claim"}

def test_failure_does_not_record_success(tmp_path):
    path = tmp_path / "jobs.sqlite"
    def fail(): raise RuntimeError("unavailable")
    with pytest.raises(RuntimeError): m.run_once(path, "job-1", fail)
    assert m.run_once(path, "job-1", lambda: {"ok": True}) == {"ok": True}

def test_sql_parameterization_and_result_contract(tmp_path):
    path = tmp_path / "jobs.sqlite"
    key = "x'); DROP TABLE jobs; --"
    assert m.run_once(path, key, lambda: {"ok": True}) == {"ok": True}
    with pytest.raises(TypeError): m.run_once(path, "second", lambda: [])
