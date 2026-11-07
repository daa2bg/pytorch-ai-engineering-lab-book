"""Execute draft worked examples in isolated Python subprocesses, recording skips.

This is an offline smoke audit, not fresh-Jupyter-kernel or editorial approval.
Fresh-kernel evidence is produced by book_factory.py validate in CI.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from book_factory import load_manifest
from validate_notebooks import code_cells

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output", default=".book-factory/audit")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(ROOT / "config/book_manifest.json")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", BOOK_RUN_DOWNLOADS="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUTF8="1", PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    # This user-supplied diagnostic hook can prevent requests/ssl imports when
    # it points outside the writable environment; do not carry it into audits.
    env.pop("SSLKEYLOGFILE", None)
    reports = []
    for chapter in manifest["chapters"]:
        if not chapter.get("managed", True):
            continue
        started = time.monotonic()
        try:
            cells = code_cells(ROOT / chapter["paths"]["notebook"])
            with tempfile.TemporaryDirectory(prefix="book-audit-") as temporary:
                script = Path(temporary) / "worked_examples.py"
                script.write_text("\n\n".join(cells), encoding="utf-8")
                result = subprocess.run([args.python, "-X", "utf8", str(script)],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
            log = result.stdout + result.stderr
            report = {"id": chapter["id"], "status": "PASS" if result.returncode == 0 else "FAIL",
                "worked_cells": len(cells), "seconds": round(time.monotonic() - started, 2),
                "skip_messages": [line for line in log.splitlines() if "SKIP" in line]}
            (output / f"{chapter['id']}.log").write_text(log, encoding="utf-8")
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report = {"id": chapter["id"], "status": "FAIL", "error": str(error)}
        reports.append(report)
        print(json.dumps(report), flush=True)
    result = {"method": "isolated offline subprocess smoke checks, not Jupyter kernels",
        "execution_python": args.python, "chapters": reports,
        "passed": sum(r["status"] == "PASS" for r in reports),
        "failed": sum(r["status"] != "PASS" for r in reports)}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
