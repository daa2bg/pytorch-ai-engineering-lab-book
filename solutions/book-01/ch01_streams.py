"""Separate Chapter 1 answers. Try each SOLO exercise before reading these."""
import json

from ai_lab.foundations import batches as my_batches


def count_epochs(records, epochs):
    """Create a new traversal each epoch; records is a reusable list of records.

    A generator object outside the loop would be exhausted after the first pass.
    Constructing the expression inside the loop makes each epoch independent.
    Empty normalized text is skipped just as in the chapter's worked pipeline.
    """
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    counts = []
    for _ in range(epochs):
        stream = (row for row in records if row["text"].strip())
        counts.append(sum(1 for _ in stream))
    return counts


def read_documents(path):
    """Yield valid JSON objects one physical line at a time.

    Skip blank lines. Invalid JSON, non-object values, and missing required keys
    raise ValueError with the physical line number. Extra fields are preserved.
    Field types/content are a later data-contract concern. File and UTF-8 errors
    propagate unchanged. Exhaustion, failure, or explicit generator.close() closes
    the file; a consumer stopping early must explicitly close the suspended stream.
    """
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected an object")
            missing = {"id", "text", "label"} - row.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing fields {sorted(missing)}")
            yield row
