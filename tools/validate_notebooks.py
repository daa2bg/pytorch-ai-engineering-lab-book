"""Execute non-exercise notebook code cells in isolated subprocesses.

Educational SOLO cells are identified by explicit metadata tags or a top-level
NotImplementedError placeholder. No arbitrary notebook from untrusted sources
should be executed by this tool: notebooks are executable code.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def exercise(cell: dict) -> bool:
    tags = cell.get("metadata", {}).get("tags", [])
    if "exercise" in tags or "skip-execution" in tags:
        return True
    source = "".join(cell.get("source", []))
    # Legacy notebooks contain SOLO cells with placeholder raises.
    return "raise NotImplementedError" in source

def code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{path}: expected notebook format 4")
    sources = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code" or exercise(cell):
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            raise ValueError(f"{path}: code cell {index} has invalid Python") from exc
        sources.append(source)
    return sources

def validate(path: Path, timeout: int) -> tuple[bool, str]:
    cells = code_cells(path)
    if not cells:
        return False, "no executable reference cells"
    # Separate subprocess for each notebook: no cross-notebook state leakage.
    script = "\n\n".join(f"# cell {i}\n{source}" for i, source in enumerate(cells))
    with tempfile.TemporaryDirectory() as temp:
        script_path = Path(temp) / "run_notebook.py"
        script_path.write_text(script, encoding="utf-8")
        env = os.environ.copy()
        # A clean kernel should import the installed package, not rely on notebook cwd.
        env["PYTHONUNBUFFERED"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)], cwd=Path.cwd(), env=env,
                text=True, capture_output=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
    if result.returncode:
        return False, (result.stderr or result.stdout)[-3500:]
    return True, f"{len(cells)} code cells executed"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="notebook paths or directories")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths] if args.paths else [Path("book/01-python-for-ai")]
    notebooks = sorted({p for path in paths for p in (
        path.rglob("*.ipynb") if path.is_dir() else [path]
    ) if p.suffix == ".ipynb" and ".ipynb_checkpoints" not in p.parts})
    if not notebooks:
        print("ERROR: no notebooks found")
        return 2
    failures = 0
    for path in notebooks:
        try:
            ok, details = validate(path, args.timeout)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            ok, details = False, str(exc)
        print(f"{'PASS' if ok else 'FAIL'} {path}: {details}")
        failures += not ok
    print(f"Validated {len(notebooks)} notebook(s); failures: {failures}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
