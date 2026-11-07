"""Durable single-worker inference jobs, not a distributed task framework.

Exactly one worker may own a database at a time. A new run treats an existing
``running`` attempt as interrupted, so callers must not use concurrent workers.
Model inference is at-least-once: a crash before the result transaction commits
can repeat the call. SQLite commits the result and successful state atomically.

predict_fn accepts the same request dictionary as the service and exposes
artifact_id and policy_version attributes (service.load_predictor provides this).
Use TransientJobError to adapt a known retryable transport failure; do not classify
every exception as transient. Each run_job call executes at most one attempt.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from time import perf_counter

from .service import validate_prediction, validate_request

MAX_ATTEMPTS = 3


class JobConflictError(ValueError):
    """The durable job ID already belongs to different input/model/policy."""


class TransientJobError(RuntimeError):
    """An adapter explicitly reports a known retryable inference/transport fault."""


class SimulatedCrash(BaseException):
    """Deliberate process-interruption seam; normal Exception handlers must not retry it."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value):
        raise ValueError(f"{name} must be a 1..128 character ASCII identifier")
    return value


def _connect(db_path):
    if str(db_path) == ":memory:":
        raise ValueError("Use a file-backed database for durable jobs")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError("Unsupported workflow database schema")
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
                artifact_id TEXT NOT NULL, policy_version TEXT NOT NULL,
                request_json TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0
                    CHECK(attempt_count BETWEEN 0 AND 3),
                status TEXT NOT NULL CHECK(status IN ('pending','running','retryable','succeeded','failed')),
                result_json TEXT, error TEXT, latency_ms REAL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                CHECK((status='succeeded' AND result_json IS NOT NULL) OR
                      (status!='succeeded' AND result_json IS NULL))
            );
            CREATE TABLE IF NOT EXISTS attempts (
                job_id TEXT NOT NULL REFERENCES jobs(job_id), number INTEGER NOT NULL,
                status TEXT NOT NULL, error TEXT, latency_ms REAL,
                started_at TEXT NOT NULL, finished_at TEXT,
                PRIMARY KEY(job_id, number)
            );
            PRAGMA user_version=1;
        """)
        return db
    except BaseException:
        db.close()
        raise


def _row(db, job_id):
    return db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()


def _public(row):
    return {key: row[key] for key in ("job_id", "input_hash", "artifact_id", "policy_version",
                                     "attempt_count", "status", "error", "latency_ms")} | {
        "result": json.loads(row["result_json"]) if row["result_json"] is not None else None}


def get_job(db_path, job_id):
    """Return redacted durable state, never the stored recovery request."""
    _identifier(job_id, "job_id")
    with closing(_connect(db_path)) as db:
        row = _row(db, job_id)
        if row is None:
            raise KeyError(job_id)
        return _public(row)


def run_job(db_path, job_id, request, predict_fn, fail_after=None):
    """Run/resume one attempt, or return a previously saved terminal job.

    request is a service request dict. Pass None only to resume from a stored
    request after restart. Model/policy identity must still match the predictor.
    Known transient exceptions leave retryable until attempt 3, then failed.
    Other ordinary exceptions are terminal; malformed inputs reject before work.

    fail_after is None, 'claim', 'inference', or 'commit'. It raises SimulatedCrash
    after that durable boundary: claim/inference leave running with no result;
    commit leaves succeeded, so the next call returns it without another inference.
    It does not kill the process. An abrupt exit during inference has the same
    persisted running/no-result state; tests also exercise a separate process.
    """
    _identifier(job_id, "job_id")
    if fail_after not in {None, "claim", "inference", "commit"}:
        raise ValueError("Unknown interruption boundary")
    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable")
    artifact_id = _identifier(getattr(predict_fn, "artifact_id", None), "artifact_id")
    policy_version = _identifier(getattr(predict_fn, "policy_version", None), "policy_version")
    clean_request = validate_request(request) if request is not None else None
    with closing(_connect(db_path)) as db:
        db.execute("BEGIN IMMEDIATE")
        row = _row(db, job_id)
        if clean_request is None:
            if row is None:
                raise ValueError("Cannot resume a job that does not exist")
            clean_request = validate_request(json.loads(row["request_json"]))
        payload = _json(clean_request)
        input_hash = sha256(payload.encode("utf-8")).hexdigest()
        if row is not None and (row["input_hash"] != input_hash or row["artifact_id"] != artifact_id or
                                row["policy_version"] != policy_version):
            raise JobConflictError("Job ID already belongs to different input/model/policy")
        if row is None:
            now = _now()
            db.execute("""INSERT INTO jobs(job_id,input_hash,artifact_id,policy_version,
                         request_json,status,created_at,updated_at) VALUES(?,?,?,?,?,'pending',?,?)""",
                       (job_id, input_hash, artifact_id, policy_version, payload, now, now))
        elif row["status"] in {"succeeded", "failed"}:
            db.commit()
            return _public(row)
        elif row["status"] == "running":
            # Caller guarantees the old worker has stopped. No leases/multiworker claim.
            db.execute("UPDATE attempts SET status='interrupted',error='interrupted',finished_at=? "
                       "WHERE job_id=? AND number=?", (_now(), job_id, row["attempt_count"]))
            db.execute("UPDATE jobs SET status='retryable',error='interrupted',updated_at=? WHERE job_id=?",
                       (_now(), job_id))
        row = _row(db, job_id)
        if row["attempt_count"] >= MAX_ATTEMPTS:
            db.execute("UPDATE jobs SET status='failed',error='attempts_exhausted',updated_at=? WHERE job_id=?",
                       (_now(), job_id))
            db.commit()
            return _public(_row(db, job_id))
        attempt = row["attempt_count"] + 1
        now = _now()
        db.execute("UPDATE jobs SET status='running',attempt_count=?,error=NULL,latency_ms=NULL,"
                   "updated_at=? WHERE job_id=?", (attempt, now, job_id))
        db.execute("INSERT INTO attempts(job_id,number,status,started_at) VALUES(?,?,'running',?)",
                   (job_id, attempt, now))
        db.commit()
        if fail_after == "claim":
            raise SimulatedCrash("Interrupted after durable claim")
        started = perf_counter()
        try:
            response = predict_fn(dict(clean_request))
            result = validate_prediction(response, request_id=clean_request["request_id"],
                                         artifact_id=artifact_id, policy_version=policy_version)
        except Exception as exc:
            transient = isinstance(exc, (TransientJobError, TimeoutError, ConnectionError))
            status = "retryable" if transient and attempt < MAX_ATTEMPTS else "failed"
            category = "transient_prediction" if transient else (
                "invalid_prediction" if isinstance(exc, (ValueError, TypeError)) else "prediction_error")
            latency = (perf_counter() - started) * 1000
            _finish(db, job_id, attempt, status, category, latency, None)
            return _public(_row(db, job_id))
        if fail_after == "inference":
            raise SimulatedCrash("Interrupted after inference, before result commit")
        latency = (perf_counter() - started) * 1000
        _finish(db, job_id, attempt, "succeeded", None, latency, _json(result))
        if fail_after == "commit":
            raise SimulatedCrash("Interrupted after durable result commit")
        return _public(_row(db, job_id))


def _finish(db, job_id, attempt, status, error, latency, result_json):
    db.execute("BEGIN IMMEDIATE")
    now = _now()
    db.execute("UPDATE jobs SET status=?,result_json=?,error=?,latency_ms=?,updated_at=? WHERE job_id=?",
               (status, result_json, error, latency, now, job_id))
    db.execute("UPDATE attempts SET status=?,error=?,latency_ms=?,finished_at=? WHERE job_id=? AND number=?",
               (status, error, latency, now, job_id, attempt))
    db.commit()


def job_report(db_path):
    """Operational evidence only: no text, recovery request, probability or input hash."""
    with closing(_connect(db_path)) as db:
        jobs = [dict(row) for row in db.execute(
            "SELECT job_id,artifact_id,policy_version,attempt_count,status,error,latency_ms FROM jobs ORDER BY job_id")]
        attempts = [dict(row) for row in db.execute(
            "SELECT job_id,number,status,error,latency_ms FROM attempts ORDER BY job_id,number")]
    return {"schema_version": 1, "job_count": len(jobs),
            "status_counts": dict(Counter(row["status"] for row in jobs)),
            "attempt_count": len(attempts),
            "failure_counts": dict(Counter(row["error"] for row in attempts if row["error"])),
            "jobs": jobs, "attempts": attempts}
