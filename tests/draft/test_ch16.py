import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("draft16", Path(__file__).parents[2] / "solutions/draft/ch16.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_boundary_and_preservation():
    payload = {"text": "Invoice"}
    assert m.predict_request(payload, lambda text: "invoice") == {"label": "invoice"}
    assert payload == {"text": "Invoice"}

@pytest.mark.parametrize("payload,error", [(None, TypeError), ({}, ValueError),
    ({"text": 1}, TypeError), ({"text": " "}, ValueError),
    ({"text": "x", "key": "secret"}, ValueError), ({"text": "x" * 201}, ValueError)])
def test_rejects_bad_requests(payload, error):
    with pytest.raises(error): m.predict_request(payload, lambda _: pytest.fail("model called"))

def test_model_contract():
    with pytest.raises(ValueError): m.predict_request({"text": "x"}, lambda _: "unexpected")
    with pytest.raises(ValueError): m.predict_request({"text": "x"}, lambda _: [])
