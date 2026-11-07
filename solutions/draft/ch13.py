"""Reference answer: cosine similarity with explicit shape and zero policies."""
import math


def cosine(left, right):
    """Equal nonempty finite real vectors; zero norm returns 0.0."""
    if len(left) != len(right) or not left:
        raise ValueError("vectors must have equal nonzero width")
    if not all(math.isfinite(x) for x in (*left, *right)):
        raise ValueError("vectors must be finite")
    left_scale = max(abs(x) for x in left)
    right_scale = max(abs(x) for x in right)
    if left_scale == 0 or right_scale == 0:
        return 0.0
    # Positive rescaling preserves direction and avoids squaring huge/tiny inputs.
    a = [x / left_scale for x in left]
    b = [x / right_scale for x in right]
    left_norm = math.sqrt(math.fsum(x * x for x in a))
    right_norm = math.sqrt(math.fsum(x * x for x in b))
    return math.fsum(x * y for x, y in zip(a, b)) / (left_norm * right_norm)
