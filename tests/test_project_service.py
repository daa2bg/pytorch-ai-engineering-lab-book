"""True JSON transport tests; tiny synthetic training fixtures are not task-quality evidence."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient
import pytest
import torch

from ai_lab.projects.sms import (load_sms_records, make_sms_splits, predict_sms,
                                 save_sms_artifact, train_sms_model)
from ai_lab.projects.service import create_app


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("service-model")
    raw = root / "messages"
    raw.write_text("\n".join([f"ham\tMeet for lunch tomorrow {i}" for i in range(16)] +
                              [f"spam\tFree cash prize today {i}" for i in range(8)]) + "\n",
                   encoding="utf-8")
    splits = make_sms_splits(load_sms_records(raw), train_size=12, validation_size=6, test_size=6)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        artifact = train_sms_model(splits, epochs=2, batch_size=4, hidden_dim=4, min_df=1)
    finally:
        torch.set_num_threads(old_threads)
    artifact.metadata["routing_policy"]["review_threshold"] = 1.0
    artifact.metadata["routing_policy"]["selection"] = "synthetic all-review boundary test"
    return save_sms_artifact(artifact, root / "bundle"), artifact


def test_lifespan_loads_once_and_http_uses_actual_saved_model(trained_bundle, monkeypatch):
    path, artifact = trained_bundle
    import ai_lab.projects.service as service
    calls = []
    original = service.load_predictor
    monkeypatch.setattr(service, "load_predictor", lambda p: (calls.append(p), original(p))[1])
    app = create_app(path)
    assert calls == []
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"ready": True, "schema_version": 1, "task_id": "sms-routing.v1",
                                 "artifact_id": artifact.metadata["artifact_id"],
                                 "policy_version": "routing-policy.v1"}
        for n in range(2):
            response = client.post("/predict", json={"request_id": f"case-{n}", "text": "Lunch tomorrow?"})
            assert response.status_code == 200
            body = response.json()
            expected = predict_sms(artifact, ["Lunch tomorrow?"])[0]
            assert body["label"] == expected["label"]
            assert body["confidence"] == pytest.approx(expected["confidence"], abs=1e-6)
            assert body["route"] == "review"
            assert list(body["probabilities"]) == ["ham", "spam"]
            assert set(body) == {"request_id", "task_id", "artifact_id", "label", "probabilities",
                                 "confidence", "route", "policy_version"}
    assert len(calls) == 1
    assert app.state.predictor is None


@pytest.mark.parametrize("payload", [
    {}, {"request_id": "a", "text": " "}, {"request_id": "a", "text": 17},
    {"request_id": 17, "text": "hello"}, {"request_id": "", "text": "hello"},
    {"request_id": "a", "text": None}, {"request_id": "a", "text": "x" * 4097},
    {"request_id": "a", "text": "hello", "artifact_path": "../other"},
    {"request_id": "a", "text": "hello", "review_threshold": 0},
    {"request_id": "line\nbreak", "text": "hello"},
])
def test_http_rejects_bad_inputs_without_echoing_text(trained_bundle, payload):
    with TestClient(create_app(trained_bundle[0])) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert response.json() == {"error": "invalid_request"}


def test_http_invalid_json_and_boundary_length(trained_bundle):
    with TestClient(create_app(trained_bundle[0])) as client:
        assert client.post("/predict", content="{", headers={"Content-Type": "application/json"}).status_code == 422
        assert client.post("/predict", json={"request_id": "limit", "text": "x" * 4096}).status_code == 200


def test_startup_failure_and_unstarted_app_never_claim_ready(tmp_path):
    app = create_app(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


def test_corrupt_model_fails_lifespan_before_readiness(trained_bundle, tmp_path):
    target = tmp_path / "corrupt"
    target.mkdir()
    for name in ("manifest.json", "preprocessing.json", "weights.pt"):
        (target / name).write_bytes((trained_bundle[0] / name).read_bytes())
    with (target / "weights.pt").open("ab") as stream:
        stream.write(b"changed")
    app = create_app(target)
    with pytest.raises(ValueError, match="checksum"):
        with TestClient(app):
            pass
    assert app.state.predictor is None


def test_no_lifespan_means_not_ready(trained_bundle):
    # Deliberately omit context manager: Starlette does not run lifespan here.
    client = TestClient(create_app(trained_bundle[0]))
    try:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"request_id": "a", "text": "Hello"}).status_code == 503
    finally:
        client.close()


def test_http_predictor_failure_is_generic_and_redacted(trained_bundle):
    app = create_app(trained_bundle[0])
    class Failing:
        artifact_id = "test-artifact"
        policy_version = "routing-policy.v1"
        def __call__(self, request):
            raise RuntimeError("private submitted message must never be echoed")
    with TestClient(app) as client:
        app.state.predictor = Failing()
        response = client.post("/predict", json={"request_id": "a", "text": "some message"})
        assert response.status_code == 500
        assert response.json() == {"detail": {"error": "prediction_failed"}}


def test_http_transient_failure_has_bounded_retry_hint(trained_bundle):
    app = create_app(trained_bundle[0])
    class Timeout:
        def __call__(self, request):
            raise TimeoutError("transport detail must not be returned")
    with TestClient(app) as client:
        app.state.predictor = Timeout()
        response = client.post("/predict", json={"request_id": "a", "text": "Hello"})
        assert response.status_code == 503 and response.headers["Retry-After"] == "1"
        assert response.json() == {"detail": {"error": "prediction_unavailable"}}


def test_loopback_fresh_process_uses_bundle_only(trained_bundle, tmp_path):
    pytest.importorskip("uvicorn", reason="Loopback transport requires the book uvicorn dependency")
    path, artifact = trained_bundle
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env.pop("SSLKEYLOGFILE", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["AI_LAB_ALLOW_DOWNLOADS"] = "0"
    env["AI_LAB_SMS_PATH"] = str(tmp_path / "missing")
    env["USE_TF"] = "0"
    # The source/trainer seams raise if touched. Model loading and actual HTTP
    # sockets remain enabled. This test never binds to a public address.
    script = """
import sys, torch
import ai_lab.projects.sms as sms
def prohibited(*args, **kwargs):
    raise AssertionError('server tried source ingestion or training')
sms.load_sms_records = prohibited
sms.train_sms_model = prohibited
torch.set_num_threads(1)
from ai_lab.projects.service import main
main(sys.argv[1:])
"""
    command = [sys.executable, "-c", script, "--bundle", str(path), "--port", str(port)]
    process = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 40
        while True:
            if process.poll() is not None:
                raise AssertionError(process.communicate()[1])
            try:
                with urlopen(base + "/health", timeout=1) as response:
                    health = json.load(response)
                break
            except (URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise AssertionError("Loopback service did not become ready")
                time.sleep(0.05)
        assert health["artifact_id"] == artifact.metadata["artifact_id"]
        request = Request(base + "/predict", data=json.dumps({"request_id": "fresh", "text": "Lunch tomorrow?"}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        expected = predict_sms(artifact, ["Lunch tomorrow?"])[0]
        assert result["label"] == expected["label"]
        assert result["confidence"] == pytest.approx(expected["confidence"], abs=1e-6)
        assert result["artifact_id"] == expected["artifact_id"]
    finally:
        process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
