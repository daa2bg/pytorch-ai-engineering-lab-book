"""Provider-independent validation of an intentionally narrow result schema."""
import json


def parse_route(payload: str, allowed_labels=("claim", "invoice", "review")) -> dict:
    if not isinstance(payload, str):
        raise TypeError("payload must be JSON text")
    result = json.loads(payload)
    if not isinstance(result, dict) or set(result) != {"label", "reason"}:
        raise ValueError("expected exactly label and reason")
    if result["label"] not in allowed_labels or not isinstance(result["reason"], str):
        raise ValueError("invalid route result")
    return result
