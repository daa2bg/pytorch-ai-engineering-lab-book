"""Reference answer: identify the content and preprocessing contract together."""
import hashlib
import json


def dataset_fingerprint(records, config):
    """Hash JSON-compatible records in order plus configuration.

    Dictionary key insertion order is irrelevant; row order matters. NaN and
    infinity are rejected rather than written as nonstandard JSON values.
    """
    payload = json.dumps(
        {"records": records, "config": config}, sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
