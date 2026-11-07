"""Reference answer: retrieval coverage measured independently of generation."""


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Unique relevant IDs recovered in the first k positions / all relevant IDs.

    Caller supplies sequences of hashable IDs and positive integer k. Duplicate
    retrieved IDs do not create extra credit. Empty relevance sets are invalid.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("at least one relevant ID is required")
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)
