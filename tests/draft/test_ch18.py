import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft18", Path(__file__).parents[2] / "solutions/draft/ch18.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def event():
    return {"request_id": "generated-1", "model_version": "v1", "latency_ms": 2,
            "decision": "accepted", "text": "private", "authorization": "secret"}

def test_allowlist_excludes_sensitive_payload():
    source = event()
    result = m.audit_event(source)
    assert set(result) == {"request_id", "model_version", "latency_ms", "decision"}
    assert "private" not in repr(result) and "secret" not in repr(result)
    assert source["text"] == "private"

@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf")])
def test_invalid_latency(latency):
    source = event(); source["latency_ms"] = latency
    with pytest.raises(ValueError): m.audit_event(source)

def test_unknown_decision():
    source = event(); source["decision"] = "anything"
    with pytest.raises(ValueError): m.audit_event(source)
    source["decision"] = []
    with pytest.raises(ValueError): m.audit_event(source)
