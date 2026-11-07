# Study 10 — Turn incoming files into a usable dataset

**First draft.** Build a small ingestion boundary that retains good records and reports bad ones. Prerequisites: dictionaries, iteration, and exceptions. This study runs offline with Python's standard library.

## Treat file parsing as a data decision

A model can train successfully on the wrong rows. Ingestion therefore needs an explicit contract before numerical processing starts. A **schema** describes expected fields and types. **Provenance** records where a row came from so a later error can be traced. **Quarantine** holds rejected-row metadata for investigation instead of silently making up replacement values.

Our minimal record has a nonblank string `id` and a string `text`. Blank text is valid at ingestion; a later task may choose to filter it. Category labels are optional here because unlabeled records are useful for inference. For supervised training, validate labels at the training-data boundary.

JSON Lines stores one JSON value per line. It is easy to stream and append, but malformed lines still require a policy. CSV stores a table of string fields; quoting matters when values contain commas or newlines. Neither format validates your business contract automatically.

## Read a tiny file without depending on the working directory

`pathlib.Path` represents filesystem paths. `with` scopes the open file's lifetime, including exception paths. Give text files an explicit encoding rather than relying on a machine default. The example creates a temporary fixture; it does not inspect real customer files.

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as folder:
    source_path = Path(folder) / "documents.jsonl"
    fixture = [
        {"id": "a", "text": "Invoice due"},
        {"id": "b", "text": "Claim received"},
    ]
    with source_path.open("w", encoding="utf-8") as handle:
        for record in fixture:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    loaded = []
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            loaded.append(json.loads(line))
    assert loaded == fixture
    print("Rows loaded:", len(loaded))
```

`enumerate(handle, 1)` pairs each line with a one-based number. We did not use `read()` to load the complete file first. The small `loaded` list is appropriate for a fixture, but a real stream can yield validated records downstream. For very large inputs, also bound record size and define how files are resumed after failure.

## Separate syntax errors from contract errors

Valid JSON can still be an unusable sample: `[]` parses successfully, and `{"id":17}` is a valid object with the wrong fields. Check the parsed value, not just whether parsing raised an exception.

```python
raw_lines = [
    '{"id":"a","text":"Invoice due"}',
    '{broken',
    '{"id":"b","text":17}',
]
usable = []
rejected = []
for line_number, line in enumerate(raw_lines, 1):
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        rejected.append({"line": line_number, "reason": "invalid JSON"})
        continue
    if not isinstance(record, dict) or not isinstance(record.get("text"), str):
        rejected.append({"line": line_number, "reason": "invalid text"})
        continue
    usable.append(record)
assert len(usable) == 1
assert [row["line"] for row in rejected] == [2, 3]
print("Accepted:", len(usable), "Rejected:", rejected)
```

This worked policy is deliberately incomplete: it does not yet validate IDs or duplicates. The exercise finishes that boundary. A rejection report should contain source identifiers, line numbers, and reason codes, not raw sensitive text. Keep the actual source separately under an appropriate retention/access policy.

Catch specific data errors. An I/O failure or programming error should not be transformed into “zero usable rows.” Otherwise the next training stage can appear successful while consuming an empty dataset.

## A CSV boundary uses the same contract

```python
import csv
from io import StringIO

csv_text = 'id,text\na,"Invoice, payment due"\nb,Claim received\n'
rows = list(csv.DictReader(StringIO(csv_text)))
assert rows[0]["text"] == "Invoice, payment due"
assert rows[1]["id"] == "b"
print(rows)
```

`DictReader` uses the header as field names. When opening an actual CSV file, use `open(..., encoding="utf-8", newline="")` so the CSV module controls newline handling. Values remain strings unless you explicitly convert them. A missing column and an empty value are different cases worth testing.

## Essential exercise — one-pass quarantine

Implement `quarantine_jsonl(lines)` returning `(accepted, rejected)`. Ignore blank lines. Accept only objects with a nonblank string ID and string text. Preserve accepted objects unchanged and in order. After the first valid occurrence, quarantine duplicate IDs using exact ID equality. A bad earlier row must not reserve its ID.

Each rejection contains exactly `line` and `reason`, with one-based line numbers. Use reasons `invalid JSON`, `expected object`, `invalid id`, `invalid text`, and `duplicate id`. The iterable supplies strings; source iteration errors propagate.

```python exercise
def quarantine_jsonl(lines):
    raise NotImplementedError("Validate each line and retain a rejection report")

good, bad = quarantine_jsonl([
    '{"id":"a","text":"ok"}', '', '{"id":"a","text":"again"}', '[]'])
assert good == [{"id": "a", "text": "ok"}]
assert bad == [{"line": 3, "reason": "duplicate id"},
               {"line": 4, "reason": "expected object"}]
```

**Optional study:** add a filename and a stable content hash to the report. Decide whether duplicate IDs across files indicate retries, legitimate revisions, or corruption; that decision belongs in an explicit versioning contract.

**Completion gate:** parse JSONL and CSV fixtures, classify malformed syntax versus wrong schema, and demonstrate that a bad row does not silently alter a good one.

Primary sources: https://docs.python.org/3.11/library/json.html and https://docs.python.org/3.11/library/csv.html . Reference answer: `solutions/draft/ch10.py`; tests: `tests/draft/test_ch10.py`.
