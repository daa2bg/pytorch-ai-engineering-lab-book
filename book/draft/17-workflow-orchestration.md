# Study 17 — A Workflow That Remembers Its Work

**First draft. Core study, about 45 minutes.** A document workflow can validate a record, prepare features, invoke a model, and save the result. A process crash should not erase every completed decision. This study introduces durable state with SQLite and uses an explicit tool registry instead of arbitrary executable instructions.

## State belongs outside the worker process

A Python dictionary is a useful cache, but it disappears when the process exits. A durable **checkpoint** stores enough information to continue after restart. A job ID identifies a logical piece of work; a completed result associated with that ID can be reused when the caller retries.

Use an actual temporary database file rather than an in-memory database for the restart experiment. Closing and reopening a connection demonstrates persistence without a database server. A `TemporaryDirectory` owns the example's files and cleans them up afterward.

```python
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as folder:
    database = Path(folder) / "workflow.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("CREATE TABLE results (job_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            connection.execute("INSERT INTO results VALUES (?, ?)",
                               ("doc-1", json.dumps({"label": "invoice"})))
    with closing(sqlite3.connect(database)) as reopened:
        row = reopened.execute("SELECT payload FROM results WHERE job_id = ?", ("doc-1",)).fetchone()
        assert json.loads(row[0]) == {"label": "invoice"}
        print("Recovered:", json.loads(row[0]))
```

The inner connection context commits on normal exit and rolls back on an exception. It does not close the connection; `closing(...)` handles that separate resource lifecycle. The question marks are parameter placeholders. Values are supplied separately, rather than concatenated into SQL, so an ID cannot turn into SQL syntax.

Our table stores a JSON result, not Python pickle data or arbitrary executable objects. JSON has a limited data model: dictionaries, lists, strings, numbers, Booleans, and null. Define a result schema before integrating multiple independent workers.

## Select tools through an explicit contract

An orchestrator chooses which operation runs next. When an LLM later suggests a tool name, that suggestion is still data. Resolve it through a trusted registry; never evaluate generated Python or shell text.

```python
def word_count(text):
    return {"word_count": len(text.split())}

tools = {"count_words": word_count}

def dispatch(name, text):
    if name not in tools:
        raise ValueError("unknown tool")
    if not isinstance(text, str) or len(text) > 1000:
        raise ValueError("tool input must be bounded text")
    return tools[name](text)

assert dispatch("count_words", "small reliable workflow") == {"word_count": 3}
print(dispatch("count_words", "Invoice due"))
```

This registry defines capabilities: the orchestrator can count words, not read arbitrary files or execute commands. A production tool would also need authorization for its data and effects. A correct tool name does not establish permission to perform a particular action.

### SOLO exercise — reuse a completed result after restart

Implement `run_once(database, job_id, handler)`. For this single-worker lab, open SQLite, create a `jobs(id TEXT PRIMARY KEY, result TEXT NOT NULL)` table if needed, and return the stored JSON dictionary if `job_id` is present. Otherwise call `handler()`, require a JSON-serializable dictionary, save it, and return its JSON-round-tripped value. Reject a blank/non-string ID with `ValueError`; a non-dictionary result raises `TypeError`. Propagate handler/serialization/database errors, and do not record a successful result when they happen. Use `json.dumps(..., allow_nan=False)` and parameterized SQL.

```python exercise
def run_once(database, job_id, handler):
    raise NotImplementedError("Implement durable result reuse")

# In your own temporary directory, call twice using the same database and ID.
# The second handler should raise if invoked; the stored result should be returned instead.
```

The function name describes this lab's normal retry behavior, not a global exactly-once guarantee. A worker can crash after an external side effect but before storing the result. The retry may perform that effect again. Likewise, two workers can both see an absent result unless you introduce a proper claim/lease protocol. Do not apply this simple single-worker pattern to concurrent payment, email, or mutation tools without stronger coordination and idempotency support.

**Completion gate:** show persistence across closed connections, a failed handler followed by a successful retry, and an ID containing quote characters. Explain the crash window in your own words. **Optional study:** add explicit `pending`, `running`, `completed`, and `failed` states, with attempt counts and a lease expiry, then reason about recovery before implementing parallel workers.

Reference answers: `solutions/draft/ch17.py`; tests: `tests/draft/test_ch17.py`. Primary source: [Python SQLite transactions, parameters, and connection management](https://docs.python.org/3.11/library/sqlite3.html). Next, record operational evidence without copying private document contents into telemetry.
