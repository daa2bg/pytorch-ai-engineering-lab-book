# Study 18 — Observe Decisions Without Logging Documents

**First draft. Core study, about 40 minutes.** A prediction service needs enough evidence to investigate failures: which model ran, how long it took, whether validation failed, and which trace connects its stages. Logging the entire document is an easy shortcut that can create a second, poorly controlled copy of sensitive data.

The deliverable is a small, allowlisted audit event and an operational summary. The same principle applies to API tokens, retrieved passages, prompts, customer metadata, and tool responses: useful telemetry is a designed data product, not a dump of every object in memory.

## Start with questions the event must answer

A **trace ID** connects events belonging to one request. A **model version** identifies the artifact/configuration used for prediction. Latency describes elapsed time, while a small decision vocabulary makes failures countable. These fields support operational investigation without needing the original text in the event.

Use a generated identifier, not an email address or customer number. Even a schema containing only `request_id` can carry personal information if the caller fills it with personal information. Field names are not a privacy guarantee.

```python
import json
from uuid import uuid4

source = {
    "request_id": str(uuid4()), "model_version": "local-classifier-v1",
    "latency_ms": 12.5, "decision": "accepted",
    "text": "A private document", "authorization": "do-not-log-this",
}
operational = {name: source[name] for name in
               ("request_id", "model_version", "latency_ms", "decision")}
encoded = json.dumps(operational, sort_keys=True)
assert "private" not in encoded
assert "do-not-log-this" not in encoded
print(encoded)
```

The dictionary comprehension selects approved fields. This is more stable than a blacklist of current secret names: a newly added payload field is excluded by default. Keep the source record unchanged so validation and model execution see the original request.

## Turn events into useful operational checks

Counts and ratios can reveal problems before someone inspects individual logs. A rising rejected-request count may indicate a schema change, bad upstream data, or misuse. It does not, by itself, show degraded model accuracy. Model quality needs evaluation against trustworthy labels, sometimes collected later.

```python
events = [
    {"decision": "accepted", "latency_ms": 8.0},
    {"decision": "rejected", "latency_ms": 1.0},
    {"decision": "accepted", "latency_ms": 10.0},
]
counts = {}
for event in events:
    decision = event["decision"]
    counts[decision] = counts.get(decision, 0) + 1
assert counts == {"accepted": 2, "rejected": 1}
print("Decision counts:", counts)
```

A drift signal describes a change in the inputs or predictions over time. It is a reason to investigate, not automatic proof that a new model is needed. Compare like periods and data sources, preserve an evaluation baseline, and distinguish changes in traffic mix from actual error changes.

Python logging can carry the serialized event to the application's configured destination. Libraries should obtain a named logger and leave output configuration to the application. Do not log entire caught exceptions blindly: error messages may contain user-provided text or credentials.

### SOLO exercise — validate and filter an audit event

Write `audit_event(event)` returning a new dictionary with exactly `request_id`, `model_version`, `latency_ms`, and `decision`. Required keys missing from the dictionary raise `KeyError`. IDs/version must be nonblank strings, latency must be a finite, nonnegative integer or float (reject Boolean values with `TypeError`), and decision must be `accepted`, `rejected`, or `failed`. Invalid values raise `ValueError`. Convert latency to float, ignore all other fields, and never mutate the source. The caller owns the promise that its identifiers contain no sensitive information.

```python exercise
def audit_event(event):
    raise NotImplementedError("Create a validated allowlisted event")

event = {"request_id": "trace-1", "model_version": "v1", "latency_ms": 3,
         "decision": "accepted", "text": "private"}
assert audit_event(event) == {"request_id": "trace-1", "model_version": "v1",
                              "latency_ms": 3.0, "decision": "accepted"}
assert event["text"] == "private"
```

Use `math.isfinite` rather than checking only `latency >= 0`: NaN and infinity are valid floating-point values but poor operational measurements. The allowlist removes raw content from this event; it does not remove that content from every log, tracing span, exception handler, or third-party monitoring agent. Review the complete data path, access controls, and retention policy when deploying the system.

**Completion gate:** prove an unexpected secret field is excluded, an invalid latency is rejected, and the source survives unchanged. Explain the difference between a service-health metric and a model-quality metric. **Optional study:** add a windowed rejection-rate alert with a minimum sample count; explain how a very small traffic window could otherwise create noisy alerts.

Reference answers: `solutions/draft/ch18.py`; tests: `tests/draft/test_ch18.py`. Primary sources: [Python logging](https://docs.python.org/3.11/library/logging.html) and [finite-number checks](https://docs.python.org/3.11/library/math.html#math.isfinite). The next optional studies return to model internals: image/sequence shapes and attention.
