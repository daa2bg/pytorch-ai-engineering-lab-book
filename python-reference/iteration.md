# Iteration and generators — 10-minute refresher

An *iterable* produces an iterator through `iter(x)`. An iterator advances via `next(it)` and remembers its state. A generator function suspends at `yield`.

```python
squares = (n*n for n in range(4))
assert list(squares) == [0, 1, 4, 9]
assert list(squares) == []  # exhausted
```

**AI connection:** streaming examples and mini-batches without loading everything into memory. A consumed generator cannot restart automatically for the next epoch.

**SOLO drill:** Implement `take(n, iterable)` that yields at most n elements, never consumes an extra item, and rejects negative n.

**Check:** Why does `len(generator)` fail? Why does converting a generator to a list consume it?

Official references: https://docs.python.org/3/tutorial/classes.html#iterators and https://docs.python.org/3/howto/functional.html
