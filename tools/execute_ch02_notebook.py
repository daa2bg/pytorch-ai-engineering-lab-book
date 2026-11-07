"""Validate and execute Chapter 2 worked examples in a fresh Jupyter kernel.

Usage: python tools/execute_ch02_notebook.py --output-dir PATH
The source notebook remains unexecuted. Evidence goes to the requested folder.
No installation of a global kernelspec is required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbclient
from nbclient import NotebookClient
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "book/01-python-for-ai/02-functions-and-data-model.ipynb"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(SOURCE, as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory(prefix="ch02-kernel-") as temporary:
        # Use an ephemeral kernelspec so no machine-wide kernelspec or user
        # configuration can silently select a different Python environment.
        kernel_dir = Path(temporary) / "python3"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Chapter 2 validation", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="python3", connection_file=str(Path(temporary) / "connection.json"),
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False),
        )
        client = NotebookClient(
            notebook, km=manager, timeout=60,
            resources={"metadata": {"path": temporary}},
            skip_cells_with_tag="exercise", allow_errors=False,
        )
        kernel_env = os.environ.copy()
        kernel_env["IPYTHONDIR"] = str(Path(temporary) / "ipython")
        kernel_env["JUPYTER_CONFIG_DIR"] = str(Path(temporary) / "jupyter-config")
        try:
            client.execute(env=kernel_env)
        finally:
            # A supplied KernelManager is owned by this runner, not nbclient.
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    skipped = [c for c in code if "exercise" in c.metadata.get("tags", [])]
    executed = [c for c in code if c.execution_count is not None]
    assert len(executed) + len(skipped) == len(code)
    assert all(c.execution_count is None and not c.outputs for c in skipped)
    output_text = "\n".join(
        output.get("text", "") for cell in executed for output in cell.outputs
        if output.output_type == "stream"
    )
    tensor_bridge = "PASS" if "PyTorch inputs:" in output_text else "SKIP"
    assert tensor_bridge == "PASS" or "SKIP tensor bridge:" in output_text
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "executable": sys.executable,
        "nbformat": nbformat.__version__, "nbclient": nbclient.__version__,
        "schema": "PASS", "fresh_kernel": "PASS", "working_directory": "temporary directory",
        "total_cells": len(notebook.cells), "code_cells": len(code),
        "executed_worked_cells": len(executed), "skipped_exercise_cells": len(skipped),
        "errors": 0,
        "tensor_bridge": tensor_bridge,
    }
    nbformat.write(notebook, args.output_dir / "ch02-executed.ipynb")
    (args.output_dir / "ch02-kernel-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
