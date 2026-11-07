"""Content fingerprints for a resumable, inspectable learning pipeline."""
import hashlib
import json


def task_fingerprint(chapter, shared_context):
    if not isinstance(chapter, dict) or not isinstance(shared_context, dict):
        raise TypeError("chapter and shared context must be dictionaries")
    canonical = json.dumps({"chapter": chapter, "shared": shared_context}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
