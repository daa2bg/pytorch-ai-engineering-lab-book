"""Reference answer: allowlisted operational telemetry without document content."""
import math


def audit_event(event: dict) -> dict:
    """Keep only approved fields; callers must supply non-sensitive trace IDs."""
    request_id, version = event["request_id"], event["model_version"]
    latency, decision = event["latency_ms"], event["decision"]
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a nonblank string")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("model_version must be a nonblank string")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise TypeError("latency_ms must be numeric")
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("latency_ms must be finite and nonnegative")
    if not isinstance(decision, str) or decision not in {"accepted", "rejected", "failed"}:
        raise ValueError("unknown decision")
    return {"request_id": request_id, "model_version": version,
            "latency_ms": float(latency), "decision": decision}
