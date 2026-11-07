# Study 25 — Learn orchestration by producing this book

**First draft; optional applied automation study.** Producing a book with coding agents is an engineering workflow: requirements become tasks, tasks produce artifacts, checks produce evidence, and an integration stage decides what can advance. It is also a useful way to study the orchestration skills used in ML pipelines.

The repository's reusable entry point is `tools/book_factory.py`; its operating guide is `docs/BOOK_FACTORY.md`. This study first demonstrates its underlying ideas with a small offline program. The example does not start agents, create threads, or call a hosted model. A real local run requires an installed and authenticated agent CLI and sufficient account usage allowance. Agents used inside this chat are a separate execution mechanism, not proof that the local CLI has been installed or tested.

## Represent dependencies as data

A **manifest** records what each task owns and requires. A **directed acyclic graph**, or DAG, describes prerequisites without circular dependencies. Some authoring can proceed in parallel once the shared outline and contracts exist. Validation and integration still depend on finished artifacts; making all stages simultaneous would only hide those dependencies.

```python
from graphlib import TopologicalSorter, CycleError

dependencies = {
    "outline": set(),
    "author-04": {"outline"},
    "author-05": {"outline"},
    "validate": {"author-04", "author-05"},
    "review": {"validate"},
}
order = list(TopologicalSorter(dependencies).static_order())
assert order.index("outline") < order.index("author-04")
assert order.index("author-05") < order.index("validate")
assert order[-1] == "review"
print("One valid stage order:", order)
```

A valid order is not necessarily unique. The two authors can run concurrently, subject to a worker limit. The important constraint is that both finish before validation starts. A cycle such as “A needs B, B needs A” should fail before spending agent time.

## Persist what was actually attempted

An in-memory `done` flag disappears when a process stops. Persist a small state record containing a schema version, selected task, requirements fingerprint, status, and evidence paths. Write to a temporary sibling and atomically replace the state file so readers do not observe a partially written JSON document.

```python
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as temporary:
    directory = Path(temporary)
    state_path = directory / "state.json"
    pending = directory / "state.pending.json"
    state = {"schema_version": 1, "chapter": "04", "status": "validated",
             "checks": {"tests": "passed", "editorial_review": "pending"}}
    pending.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    os.replace(pending, state_path)
    restored = json.loads(state_path.read_text(encoding="utf-8"))
    assert restored["status"] == "validated"
    assert restored["checks"]["editorial_review"] == "pending"
print("State survived a fresh read; review is still pending.")
```

Atomic replacement helps preserve a complete state record; it does not make every artifact write transactional or guarantee durability after every possible machine failure. A resumed run should verify artifact presence and content fingerprints before trusting earlier success. “The process exited with zero” and “the chapter meets the learning objective” are different claims.

## SOLO exercise — Decide whether cached work still applies

Implement `task_fingerprint(chapter, shared_context)`. Both inputs must be dictionaries, otherwise raise `TypeError`. Serialize an object containing those two values with sorted dictionary keys, UTF-8, and compact separators, then return its SHA-256 hexadecimal digest. Preserve list order because ordered objectives may carry meaning. Values must be JSON-serializable; standard serialization errors may propagate.

```python exercise
def task_fingerprint(chapter, shared_context):
    raise NotImplementedError("Fingerprint task and shared requirements")

assert task_fingerprint({"id":"04", "topic":"tensor"}, {"style":"short"}) == task_fingerprint({"topic":"tensor", "id":"04"}, {"style":"short"})
assert task_fingerprint({"id":"04"}, {"style":"short"}) != task_fingerprint({"id":"04"}, {"style":"long"})
```

A hash detects changed inputs, not changed meaning with certainty. Include the actual guidance text or its content digest, not only its filename. If a dependency's exported interface changes, invalidate its dependants according to the dependency contract. Do not regenerate unrelated chapters merely because another chapter's punctuation changed.

## From the exercise to the actual factory

Use `python tools/book_factory.py plan --chapter 04` to inspect a chapter assignment before dispatch. The manifest defines exact manuscript, notebook, solution, and test paths. Plan output should make the selected task and requirements visible. Consult the guide for full-run, request-routing, validation, status, and resume options; these are runnable tooling instructions, not an invitation to bypass its checks.

Git worktrees give authors separate checkouts while sharing repository history. Ownership still matters: an isolated author could edit a shared guide, so the runner must compare changed paths with the task's allowlist. Preserve failed worktrees and logs for diagnosis. An author should not approve its own prose merely because tests pass. A separate review examines explanations, exercise difficulty, correctness, and scope before candidate integration. Promotion or merging remains an explicit integration decision.

**Completion gate:** inspect a plan, reproduce a dependency cycle failure, show a state record surviving a reload, and demonstrate fingerprint invalidation after a changed requirement. Reference: `solutions/draft/ch25.py`; tests: `tests/draft/test_ch25.py`. Optional extension: use a fake CLI command to force author failure and verify that the pipeline records failure without promoting its output.

Primary references: [Python graphlib](https://docs.python.org/3/library/graphlib.html), [Git worktrees](https://git-scm.com/docs/git-worktree), [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive), and [Python os.replace](https://docs.python.org/3/library/os.html#os.replace).
