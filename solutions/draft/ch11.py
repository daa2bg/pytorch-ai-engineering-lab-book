"""Reference answer: parameterized extraction of a local training feature table."""


def joined_training_rows(connection, *, min_words=1):
    """Return id/word_count/label tuples ordered by id from the stated schema.

    documents and labels each have a unique id. Missing labels are excluded by
    the inner join. The caller supplies an integer minimum, which must be >=0.
    The connection belongs to the caller and is not closed here.
    """
    if min_words < 0:
        raise ValueError("min_words must be nonnegative")
    return connection.execute(
        "SELECT d.id, d.word_count, l.label "
        "FROM documents AS d JOIN labels AS l ON d.id = l.id "
        "WHERE d.word_count >= ? ORDER BY d.id", (min_words,)
    ).fetchall()
