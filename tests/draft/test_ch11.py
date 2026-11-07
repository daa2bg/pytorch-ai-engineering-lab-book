import importlib.util
from pathlib import Path
import sqlite3
import pytest

spec = importlib.util.spec_from_file_location("draft11", Path(__file__).resolve().parents[2] / "solutions/draft/ch11.py")
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


def test_query_excludes_unlabeled_and_preserves_order():
    con = sqlite3.connect(":memory:")
    try:
        con.executescript("CREATE TABLE documents(id TEXT PRIMARY KEY, word_count INTEGER); CREATE TABLE labels(id TEXT PRIMARY KEY, label TEXT);")
        con.executemany("INSERT INTO documents VALUES (?, ?)", [("b", 2), ("a", 5), ("c", 9)])
        con.executemany("INSERT INTO labels VALUES (?, ?)", [("b", "invoice"), ("a", "claim")])
        assert answer.joined_training_rows(con) == [("a", 5, "claim"), ("b", 2, "invoice")]
        assert answer.joined_training_rows(con, min_words=3) == [("a", 5, "claim")]
        assert con.execute("SELECT 1").fetchone() == (1,)
        with pytest.raises(ValueError):
            answer.joined_training_rows(con, min_words=-1)
    finally:
        con.close()
