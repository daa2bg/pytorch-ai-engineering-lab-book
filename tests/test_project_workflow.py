"""Synthetic predictors isolate job durability; actual model evidence is reported separately."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from ai_lab.projects.workflow import (JobConflictError, SimulatedCrash, TransientJobError,
                                      get_job, job_report, run_job)


class Predictor:
    artifact_id = "synthetic-model-v1"
    policy_version = "routing-policy.v1"

    def __init__(self):
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return {"request_id": request["request_id"], "task_id": "sms-routing.v1",
                "artifact_id": self.artifact_id, "policy_version": self.policy_version,
                "label": "ham", "probabilities": {"ham": 0.75, "spam": 0.25},
                "confidence": 0.75, "route": "review"}


@pytest.fixture
def request_data():
    return {"request_id": "request-1", "text": "private fixture should stay in the local recovery store"}


def test_success_is_durable_idempotent_and_report_does_not_leak_text(tmp_path, request_data):
    db = tmp_path / "jobs.sqlite"
    predictor = Predictor()
    first = run_job(db, "job-1", request_data, predictor)
    assert first["status"] == "succeeded" and first["attempt_count"] == 1
    assert run_job(db, "job-1", dict(request_data), predictor) == first
    assert predictor.calls == 1
    assert get_job(db, "job-1") == first
    assert request_data["text"] not in json.dumps(first)
    report = job_report(db)
    assert report["status_counts"] == {"succeeded": 1}
    assert report["attempt_count"] == 1
    assert report["jobs"][0]["latency_ms"] >= 0
    assert request_data["text"] not in json.dumps(report)
    assert "input_hash" not in json.dumps(report)
    with sqlite3.connect(db) as connection:
        stored = json.loads(connection.execute("SELECT request_json FROM jobs").fetchone()[0])
    assert stored == request_data


def test_same_job_id_rejects_changed_input_or_model_or_policy(tmp_path, request_data):
    db = tmp_path / "jobs.sqlite"
    predictor = Predictor()
    run_job(db, "job-1", request_data, predictor)
    with pytest.raises(JobConflictError):
        run_job(db, "job-1", {**request_data, "text": "different"}, predictor)
    predictor.artifact_id = "model-v2"
    with pytest.raises(JobConflictError):
        run_job(db, "job-1", request_data, predictor)
    predictor.artifact_id = "synthetic-model-v1"
    predictor.policy_version = "routing-policy.v2"
    with pytest.raises(JobConflictError):
        run_job(db, "job-1", request_data, predictor)
    assert predictor.calls == 1


@pytest.mark.parametrize("boundary,expected_calls", [("claim", 1), ("inference", 2), ("commit", 1)])
def test_interruption_boundaries_and_restart_from_saved_request(tmp_path, request_data, boundary, expected_calls):
    db = tmp_path / "jobs.sqlite"
    predictor = Predictor()
    with pytest.raises(SimulatedCrash):
        run_job(db, "job-1", request_data, predictor, fail_after=boundary)
    interrupted = get_job(db, "job-1")
    assert interrupted["status"] == ("succeeded" if boundary == "commit" else "running")
    assert (interrupted["result"] is None) == (boundary != "commit")
    resumed = run_job(db, "job-1", None, predictor)
    assert resumed["status"] == "succeeded"
    assert predictor.calls == expected_calls
    assert resumed["attempt_count"] == (1 if boundary == "commit" else 2)
    assert run_job(db, "job-1", None, predictor) == resumed
    assert predictor.calls == expected_calls


def test_only_transient_failures_retry_and_attempts_are_bounded(tmp_path, request_data):
    class Transient(Predictor):
        def __call__(self, request):
            self.calls += 1
            raise TransientJobError("private message and stack trace must not be saved")
    predictor = Transient()
    db = tmp_path / "jobs.sqlite"
    assert run_job(db, "job-1", request_data, predictor)["status"] == "retryable"
    assert run_job(db, "job-1", None, predictor)["status"] == "retryable"
    final = run_job(db, "job-1", None, predictor)
    assert final["status"] == "failed" and final["attempt_count"] == 3
    assert run_job(db, "job-1", None, predictor) == final
    assert predictor.calls == 3
    assert job_report(db)["failure_counts"] == {"transient_prediction": 3}
    assert "private message" not in json.dumps(job_report(db))


def test_known_transient_error_can_recover(tmp_path, request_data):
    class Once(Predictor):
        def __call__(self, request):
            if self.calls == 0:
                self.calls += 1
                raise TimeoutError("network timeout")
            return super().__call__(request)
    predictor = Once()
    db = tmp_path / "jobs.sqlite"
    assert run_job(db, "job-1", request_data, predictor)["status"] == "retryable"
    assert run_job(db, "job-1", None, predictor)["status"] == "succeeded"
    assert predictor.calls == 2


def test_permanent_failure_and_invalid_prediction_are_not_retried(tmp_path, request_data):
    class Broken(Predictor):
        def __call__(self, request):
            self.calls += 1
            raise RuntimeError("broken model is not a transient transport fault")
    predictor = Broken()
    db = tmp_path / "jobs.sqlite"
    assert run_job(db, "job-1", request_data, predictor)["error"] == "prediction_error"
    assert run_job(db, "job-1", None, predictor)["status"] == "failed"
    assert predictor.calls == 1
    class WrongVersion(Predictor):
        def __call__(self, request):
            return {**super().__call__(request), "artifact_id": "silently-changed-model"}
    assert run_job(db, "job-2", request_data, WrongVersion())["error"] == "invalid_prediction"


def test_schema_failures_never_run_and_missing_resume_rejects(tmp_path):
    predictor = Predictor()
    db = tmp_path / "jobs.sqlite"
    with pytest.raises(ValueError):
        run_job(db, "job-1", {"request_id": "a", "text": 12}, predictor)
    assert predictor.calls == 0 and not db.exists()
    with pytest.raises(ValueError, match="does not exist"):
        run_job(db, "missing", None, predictor)
    with pytest.raises(ValueError):
        run_job(db, "job-1", {"request_id": "a", "text": "x"}, predictor, fail_after="unknown")


def test_three_interrupted_attempts_exhaust_budget(tmp_path, request_data):
    db = tmp_path / "jobs.sqlite"
    predictor = Predictor()
    for _ in range(3):
        with pytest.raises(SimulatedCrash):
            run_job(db, "job-1", request_data, predictor, fail_after="inference")
    exhausted = run_job(db, "job-1", None, predictor)
    assert exhausted["status"] == "failed" and exhausted["error"] == "attempts_exhausted"
    assert predictor.calls == 3


def test_abrupt_process_exit_leaves_claim_and_recovery_repeats_inference(tmp_path, request_data):
    db = tmp_path / "jobs.sqlite"
    marker = tmp_path / "inference-completed.txt"
    script = """
import json, os, sys
from pathlib import Path
from ai_lab.projects.workflow import run_job
class CrashAfterInference:
    artifact_id = 'synthetic-model-v1'
    policy_version = 'routing-policy.v1'
    def __call__(self, request):
        # Stand in for an external inference completing before our commit.
        Path(sys.argv[2]).write_text('inference completed', encoding='utf-8')
        os._exit(73)
run_job(sys.argv[1], 'job-1', json.loads(sys.argv[3]), CrashAfterInference())
"""
    env = os.environ.copy()
    env.pop("SSLKEYLOGFILE", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.run([sys.executable, "-c", script, str(db), str(marker), json.dumps(request_data)],
                           cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert child.returncode == 73, child.stderr
    assert marker.is_file()
    assert get_job(db, "job-1")["status"] == "running"
    assert get_job(db, "job-1")["result"] is None
    predictor = Predictor()
    restored = run_job(db, "job-1", None, predictor)
    assert restored["status"] == "succeeded" and restored["attempt_count"] == 2
    assert predictor.calls == 1
    assert job_report(db)["failure_counts"] == {"interrupted": 1}
