"""Reference answer: retain usable JSONL rows and a small quarantine report."""
import json


def quarantine_jsonl(lines):
    """Return (records, rejected) for one pass over JSONL strings.

    Blank lines are ignored. Objects need nonblank string id and string text.
    Duplicate IDs are rejected after the first valid occurrence. Rejections
    contain one-based line and a reason, not the potentially sensitive text.
    Source iteration errors propagate; only data errors are quarantined.
    """
    accepted, rejected, seen = [], [], set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            rejected.append({"line": number, "reason": "invalid JSON"})
            continue
        if not isinstance(record, dict):
            reason = "expected object"
        elif not isinstance(record.get("id"), str) or not record["id"].strip():
            reason = "invalid id"
        elif not isinstance(record.get("text"), str):
            reason = "invalid text"
        elif record["id"] in seen:
            reason = "duplicate id"
        else:
            accepted.append(record)
            seen.add(record["id"])
            continue
        rejected.append({"line": number, "reason": reason})
    return accepted, rejected
