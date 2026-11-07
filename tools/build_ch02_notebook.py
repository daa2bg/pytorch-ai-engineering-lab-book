"""Build the complete Chapter 2 notebook from its print manuscript.

Every prose block is retained, including exercises and references. Python fences
become runnable cells; ``python exercise`` fences gain an execution-skip tag.
Only the standard library is needed to rebuild the unexecuted notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "book/01-python-for-ai/02-functions-and-data-model.md"
NOTEBOOK = MANUSCRIPT.with_suffix(".ipynb")
FENCE = re.compile(r"^```python(?P<exercise> exercise)?\n(?P<code>.*?)^```\s*$", re.M | re.S)


def build_notebook(text: str) -> dict:
    cells = []

    def add(kind: str, source: str, exercise: bool = False) -> None:
        source = source.strip() + "\n"
        if not source.strip():
            return
        identity = hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:12]
        cell = {
            "cell_type": kind,
            "id": identity,
            "metadata": {"tags": ["exercise"]} if exercise else {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    position = 0
    for match in FENCE.finditer(text):
        add("markdown", text[position:match.start()])
        add("code", match["code"], exercise=bool(match["exercise"]))
        position = match.end()
    add("markdown", text[position:])
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "chapter_source": MANUSCRIPT.relative_to(ROOT).as_posix(),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the notebook differs")
    args = parser.parse_args()
    built = build_notebook(MANUSCRIPT.read_text(encoding="utf-8"))
    if args.check:
        actual = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        if actual != built:
            print("FAIL: Chapter 2 notebook differs; rebuild it from the manuscript")
            return 1
        print(f"PASS: manuscript and complete notebook agree ({len(built['cells'])} cells)")
        return 0
    NOTEBOOK.write_text(json.dumps(built, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK.relative_to(ROOT)}: {len(built['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
