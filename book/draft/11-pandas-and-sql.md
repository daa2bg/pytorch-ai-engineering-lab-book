# Study 11 — Join and inspect data before training

**First draft.** Create a small feature table from SQLite and pandas, checking what a join does to row counts. Prerequisites: the ingestion contract and basic Python collections. SQLite is local and serverless; pandas is an optional installed dependency for the second worked section.

## A join can change the experiment

Documents and labels often arrive from different systems. A document table might contain word counts while a reviewer table contains the known category. A **join** combines rows using a shared key. If either side contains duplicate keys, the join can produce more rows than expected. Repeating a sample can change its influence during training.

Before joining, state the expected **cardinality**: one-to-one, one-to-many, or many-to-many. Our fixture expects one label per document ID. We also want to know which documents lack labels instead of quietly treating them as negative examples.

SQLite lets us practice the real SQL boundary without configuring a database server. Parameter binding keeps values separate from SQL syntax. It also avoids quoting errors when a value contains an apostrophe.

## Read a controlled local fixture

```python
import sqlite3

connection = sqlite3.connect(":memory:")
try:
    connection.executescript(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, word_count INTEGER);"
        "CREATE TABLE labels(id TEXT PRIMARY KEY, label TEXT);"
    )
    connection.executemany("INSERT INTO documents VALUES (?, ?)",
                           [("a", 5), ("b", 2), ("c", 9)])
    connection.executemany("INSERT INTO labels VALUES (?, ?)",
                           [("a", "claim"), ("b", "invoice")])
    selected = connection.execute(
        "SELECT id, word_count FROM documents WHERE word_count >= ? ORDER BY id",
        (3,),
    ).fetchall()
    assert selected == [("a", 5), ("c", 9)]
    print("Parameterized query:", selected)
finally:
    connection.close()
```

The tuple `(3,)` contains one bound value; the trailing comma creates a one-item tuple. The `?` placeholder is for a value, not a table or column name. Choose SQL identifiers from trusted code when necessary rather than interpolating arbitrary input.

A primary key enforces uniqueness in this fixture. If your production source has historical versions, an ID alone may not be a unique row key. Select the intended version before joining; `drop_duplicates` without a reason can discard useful information.

## Inspect a pandas join

A pandas **DataFrame** is a table with named columns. Readable inspection comes before feature engineering. Use a left join to retain documents while examining missing labels; then select the labeled rows deliberately.

```python
import importlib.util

if importlib.util.find_spec("pandas") is None:
    print("SKIP pandas join: pandas is not installed.")
else:
    import pandas as pd
    documents = pd.DataFrame({"id": ["a", "b", "c"], "word_count": [5, 2, 9]})
    labels = pd.DataFrame({"id": ["a", "b"], "label": ["claim", "invoice"]})
    inspected = documents.merge(labels, on="id", how="left",
                                 validate="one_to_one", indicator=True)
    missing_ids = inspected.loc[inspected["label"].isna(), "id"].tolist()
    assert missing_ids == ["c"]
    training = inspected.loc[inspected["label"].notna(),
                             ["id", "word_count", "label"]].copy()
    assert training["id"].tolist() == ["a", "b"]
    assert len(training) == 2
    print(training)
```

`validate="one_to_one"` turns an unexpected duplicate-key relationship into an error. `indicator=True` adds a column showing where each output row came from. `.loc[condition, columns]` selects rows and columns by explicit conditions. `.copy()` makes the editable table boundary clear.

A left join does not automatically mean “safe”: inspect null join keys, duplicates, unmatched rows, and source filtering. Pandas can match null keys in ways that differ from usual SQL null semantics. This fixture deliberately uses non-null IDs. Reject or explicitly handle missing identifiers before joining.

Keep split membership visible too. If a customer has several related documents, joining customer metadata does not make those examples independent. Fit numerical transformations using the training partition, and preserve group-based separation when the task requires it.

## Essential exercise — extract the agreed training rows

Implement `joined_training_rows(connection, *, min_words=1)` for the SQLite schema above. Return `(id, word_count, label)` tuples ordered by ID, using an inner join and a bound minimum. Exclude documents without labels and rows below the minimum. Reject a negative minimum with `ValueError`; the caller supplies an integer. Leave the supplied connection open.

```python exercise
def joined_training_rows(connection, *, min_words=1):
    raise NotImplementedError("Join by ID and bind the filter value")

con = sqlite3.connect(":memory:")
try:
    con.executescript(
        "CREATE TABLE documents(id TEXT PRIMARY KEY, word_count INTEGER);"
        "CREATE TABLE labels(id TEXT PRIMARY KEY, label TEXT);")
    con.execute("INSERT INTO documents VALUES (?, ?)", ("a", 5))
    con.execute("INSERT INTO labels VALUES (?, ?)", ("a", "claim"))
    assert joined_training_rows(con, min_words=3) == [("a", 5, "claim")]
finally:
    con.close()
```

**Optional study:** insert a duplicate label into a pandas fixture and verify the cardinality check fails. Compare the row count with an unchecked many-to-many merge.

**Completion gate:** explain the query placeholders, identify the unlabeled document, and demonstrate that filtering and joining preserve the intended training population.

Primary sources: https://docs.python.org/3.11/library/sqlite3.html and https://pandas.pydata.org/docs/user_guide/merging.html . Reference answer: `solutions/draft/ch11.py`; tests: `tests/draft/test_ch11.py`.
