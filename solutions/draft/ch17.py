"""Reference answer: durable memoization for a single local workflow worker."""
from contextlib import closing
import json
import sqlite3


def run_once(database, job_id: str, handler) -> dict:
    """Cache a JSON result durably. Not an exactly-once external side-effect API."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a nonblank string")
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, result TEXT NOT NULL)")
            row = connection.execute("SELECT result FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is not None:
                return json.loads(row[0])
            result = handler()
            if not isinstance(result, dict):
                raise TypeError("handler must return a JSON dictionary")
            encoded = json.dumps(result, allow_nan=False)
            connection.execute("INSERT INTO jobs(id, result) VALUES (?, ?)", (job_id, encoded))
            return json.loads(encoded)
