"""A local, review-first chapter production pipeline (Python 3.11+).

Plan creates prompts without AI. Run uses isolated Git worktrees and a configured
agent command. Nothing is pushed or merged into the source checkout. Generated
notebooks and agents are executable code: use this only with trusted repositories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

REQUIRED_PATHS = ("manuscript", "notebook", "solutions", "tests")
FENCE = re.compile(r"^```python(?P<exercise> exercise)?\n(?P<code>.*?)^```[ \t]*$", re.M | re.S)
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
TERMINAL = {"reviewed", "integrated"}


def relative_path(value: str) -> str:
    """Reject absolute, traversal, control, Git, and platform-specific paths."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"unsafe repository path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {".", "..", ".git"} for p in value.split("/")):
        raise ValueError(f"unsafe repository path: {value!r}")
    if any(ord(c) < 32 for c in value) or "//" in value:
        raise ValueError(f"unsafe repository path: {value!r}")
    return path.as_posix()


def inside(root: Path, value: str) -> Path:
    target = root / relative_path(value)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes repository through symlink: {value}")
    return target


def validate_manifest(data: dict) -> dict:
    if data.get("version") != 1 or not isinstance(data.get("chapters"), list):
        raise ValueError("manifest requires version 1 and a chapters list")
    ids, owners = set(), {}
    for chapter in data["chapters"]:
        identifier = chapter.get("id")
        if not isinstance(identifier, str) or not SAFE_ID.fullmatch(identifier) or identifier in ids:
            raise ValueError(f"invalid or duplicate chapter ID: {identifier!r}")
        ids.add(identifier)
        if chapter.get("tier") not in {"core", "study"}:
            raise ValueError(f"{identifier}: tier must be core or study")
        if not isinstance(chapter.get("managed", True), bool):
            raise ValueError(f"{identifier}: managed must be Boolean")
        for field in ("slug", "title"):
            if not isinstance(chapter.get(field), str) or not chapter[field].strip():
                raise ValueError(f"{identifier}: {field} is required")
        for field in ("objectives", "topics", "depends_on"):
            values = chapter.get(field)
            if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
                raise ValueError(f"{identifier}: {field} must be a string list")
        paths = chapter.get("paths", {})
        if any(key not in paths for key in REQUIRED_PATHS):
            raise ValueError(f"{identifier}: require paths {REQUIRED_PATHS}")
        for name in REQUIRED_PATHS:
            path = relative_path(paths[name])
            if path in owners:
                raise ValueError(f"shared artifact path: {path} ({owners[path]}, {identifier})")
            owners[path] = identifier
        suffixes = {"manuscript": ".md", "notebook": ".ipynb", "solutions": ".py", "tests": ".py"}
        if any(not paths[k].endswith(v) for k, v in suffixes.items()):
            raise ValueError(f"{identifier}: artifact extension does not match its role")
    graph = {c["id"]: c["depends_on"] for c in data["chapters"]}
    visiting, done = set(), set()

    def visit(identifier):
        if identifier in visiting:
            raise ValueError("chapter dependency cycle")
        if identifier in done:
            return
        visiting.add(identifier)
        for dependency in graph[identifier]:
            if dependency not in graph:
                raise ValueError(f"{identifier}: unknown dependency {dependency}")
            visit(dependency)
        visiting.remove(identifier)
        done.add(identifier)

    for identifier in graph:
        visit(identifier)
    for path in data.get("shared_context", []):
        relative_path(path)
    return data


def load_manifest(path: Path) -> dict:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def select_chapters(manifest: dict, chapter_ids: list[str] | None = None,
                    request: str | None = None, downstream: bool = False) -> list[dict]:
    """Explicit IDs or deterministic topic matching; never guess an empty match."""
    chapters = {c["id"]: c for c in manifest["chapters"]}
    if chapter_ids:
        missing = set(chapter_ids) - chapters.keys()
        if missing:
            raise ValueError(f"unknown chapter IDs: {sorted(missing)}")
        selected = set(chapter_ids)
        protected = [i for i in selected if not chapters[i].get("managed", True)]
        if protected:
            raise ValueError(f"protected references cannot be authored: {protected}")
    elif request:
        words = set(re.findall(r"[a-z0-9]+", request.lower())) - {
            "a", "an", "the", "add", "change", "update", "improve", "and", "of", "to", "chapter", "with", "for",
        }
        selected = set()
        for identifier, chapter in chapters.items():
            haystack = " ".join([chapter["title"], chapter["slug"], *chapter["topics"]]).lower()
            if words.intersection(re.findall(r"[a-z0-9]+", haystack)) and chapter.get("managed", True):
                selected.add(identifier)
        if not selected:
            raise ValueError("request matched no managed chapters; use explicit --chapter IDs")
    else:
        selected = {i for i, c in chapters.items() if c.get("managed", True)}
    if downstream:
        while True:
            extra = {i for i, c in chapters.items() if c.get("managed", True)
                     and set(c["depends_on"]).intersection(selected)}
            if extra <= selected:
                break
            selected |= extra
    return [c for c in manifest["chapters"] if c["id"] in selected]


def build_notebook(markdown: str, source_path: str = "manuscript.md") -> dict:
    """Retain all prose; convert Python fences and tag unfinished exercises."""
    cells = []

    def add(kind, text, exercise=False):
        text = text.strip()
        if not text:
            return
        text += "\n"
        cell = {
            "cell_type": kind,
            "id": hashlib.sha256(f"{len(cells)}:{text}".encode()).hexdigest()[:12],
            "metadata": {"tags": ["exercise"]} if exercise else {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    position = 0
    for match in FENCE.finditer(markdown):
        add("markdown", markdown[position:match.start()])
        add("code", match["code"], bool(match["exercise"]))
        position = match.end()
    add("markdown", markdown[position:])
    return {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "chapter_source": source_path,
        },
    }


def author_prompt(manifest: dict, chapter: dict, request: str = "") -> str:
    references = "\n".join(f"- {p}" for p in manifest.get("shared_context", []))
    ownership = "\n".join(f"- {k}: {chapter['paths'][k]}" for k in REQUIRED_PATHS)
    dependencies = [c for c in manifest["chapters"] if c["id"] in chapter["depends_on"]]
    dependency_context = [{"id": c["id"], "title": c["title"], "objectives": c["objectives"],
                           "paths": c["paths"]} for c in dependencies]
    return f"""Draft/revise chapter {chapter['id']}: {chapter['title']}.
Read these canonical sources first (and their applicable AGENTS.md instructions):
{references}
Read the approved Chapter 1 and current Chapter 2 as presentation references,
not minimum length targets. Read existing chapter artifacts before revising.
Objectives: {json.dumps(chapter['objectives'])}
Topics: {json.dumps(chapter['topics'])}
Prerequisites: {json.dumps(chapter['depends_on'])}. Tier: {chapter['tier']}.
Expected artifact: {chapter.get('artifact', 'A focused chapter implementation and learner exercise.')}
Primary source URLs to verify: {json.dumps(chapter.get('sources', []))}
Prerequisite contracts/artifact paths: {json.dumps(dependency_context)}
Read prerequisite artifacts when present in this snapshot. If a sibling draft is
not present, use its stated objectives as the interface; do not invent or import
unpublished sibling implementations. Keep this chapter's examples self-contained.
Change request: {request or 'Produce a focused first draft for the existing outline.'}

You own EXACTLY these four files; do not edit any other repository files:
{ownership}
Do not commit, push, merge, change branches, install global tools, or modify
shared guidance. This is a chapter worktree; preserve other work.
Use brief context -> concept -> runnable code -> learner modification -> checks.
Experienced engineers know programming but may have Python/ML gaps: explain
relevant syntax inline. Get to useful ML work quickly; no novel or API dump.
Create a print Markdown manuscript and one complete notebook containing ALL
prose/examples/prompts. Use ```python for worked code and ```python exercise
for genuinely unfinished learner code. Keep answers in the separate solution
file; keep meaningful contract tests in the test file. No paid services for
required practice. Make downloads, optional dependencies and skipped work explicit.
The pipeline regenerates the notebook deterministically from the manuscript,
so do not add notebook-only content. Worked examples must not depend on learner
answers. Use small CPU-friendly examples, primary source links, clear contracts,
and a short completion gate. Do not mark a first draft release-ready.
Report changed files, exact checks/results, sources, and anything untested.
"""


def review_prompt(chapter: dict) -> str:
    return f"""Independently review chapter {chapter['id']}: {chapter['title']}.
Read BOOK_GOALS.md, BOOK_STYLE_GUIDE.md, CURRICULUM.md and the approved Chapter 1.
Read these artifacts: {json.dumps(chapter['paths'])}.
This is READ-ONLY: do not change any file, run Git mutations, or modify branches.
Check focused scope, Python/ML correctness, progressive explanation, exercise
contracts/solutions, meaningful tests, source quality, and complete manuscript/
notebook parity. Separate mandatory fixes from optional enrichment. Passing tests
do not establish pedagogy quality or book-wide release readiness.
Your FINAL response must be ONLY JSON:
{{"approved": true or false, "findings": ["concrete finding"], "summary": "brief assessment"}}
Approve only if no mandatory issues remain. Missing artifacts or unverified
important behavior must be reported, not silently accepted.
"""


def git(root: Path, *arguments: str, check=True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *arguments], cwd=root, text=True, encoding="utf-8",
                            errors="replace", capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result


def changed_paths(root: Path) -> set[str]:
    tracked = git(root, "diff", "--no-renames", "--name-only", "-z", "HEAD").stdout
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout
    return set(filter(None, (tracked + untracked).split("\0")))


def check_scope(root: Path, chapter: dict) -> None:
    allowed = set(chapter["paths"].values())
    extra = changed_paths(root) - allowed
    if extra:
        raise ValueError(f"agent modified files outside chapter ownership: {sorted(extra)}")
    for relative in allowed:
        inside(root, relative)


def artifact_hashes(root: Path, chapter: dict) -> dict:
    return {p: hashlib.sha256(inside(root, p).read_bytes()).hexdigest()
            for p in chapter["paths"].values() if inside(root, p).is_file()}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_chapter(root: Path, chapter: dict) -> None:
    paths = chapter["paths"]
    manuscript = inside(root, paths["manuscript"])
    notebook = inside(root, paths["notebook"])
    notebook.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(notebook, build_notebook(manuscript.read_text(encoding="utf-8"), paths["manuscript"]))


def load_chapter_notebook(root: Path, chapter: dict):
    """Validate raw manuscript parity before nbformat normalizes cell sources."""
    import nbformat
    paths = chapter["paths"]
    for name in REQUIRED_PATHS:
        if not inside(root, paths[name]).is_file():
            raise ValueError(f"missing {name}: {paths[name]}")
    expected = build_notebook(inside(root, paths["manuscript"]).read_text(encoding="utf-8"), paths["manuscript"])
    actual = json.loads(inside(root, paths["notebook"]).read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("notebook differs from manuscript; run build first")
    notebook = nbformat.read(inside(root, paths["notebook"]), as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    if not code or not any("exercise" not in c.metadata.get("tags", []) for c in code):
        raise ValueError("chapter has no executable worked code")
    for cell in code:
        if "raise NotImplementedError" in cell.source and "exercise" not in cell.metadata.get("tags", []):
            raise ValueError("unfinished learner code must have the exercise tag")
    return notebook


def validate_chapter(root: Path, chapter: dict, output: Path, timeout: int = 180,
                     execution_python: str | None = None) -> dict:
    """Schema, exact parity, fresh-kernel worked cells and separate pytest tests."""
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    paths = chapter["paths"]
    notebook = load_chapter_notebook(root, chapter)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    execution_python = execution_python or sys.executable
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root), env.get("PYTHONPATH", "")])
    result = subprocess.run([execution_python, "-m", "pytest", "-q", paths["tests"]], cwd=root,
                            text=True, capture_output=True, timeout=timeout, env=env, check=False)
    (output / "pytest.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"reference tests failed (exit {result.returncode}); see {output / 'pytest.log'}")
    with tempfile.TemporaryDirectory(prefix="book-kernel-") as temporary:
        kernel = Path(temporary) / "python3"
        kernel.mkdir()
        atomic_json(kernel / "kernel.json", {
            "argv": [execution_python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Book validation", "language": "python",
        })
        manager = KernelManager(kernel_name="python3", connection_file=str(Path(temporary) / "connection.json"),
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary], ensure_native_kernel=False))
        client = NotebookClient(notebook, km=manager, timeout=timeout, startup_timeout=60,
                                skip_cells_with_tag="exercise", allow_errors=False,
                                resources={"metadata": {"path": str(root)}})
        kernel_env = env.copy()
        kernel_env["IPYTHONDIR"] = str(Path(temporary) / "ipython")
        kernel_env["JUPYTER_CONFIG_DIR"] = str(Path(temporary) / "jupyter")
        try:
            client.execute(env=kernel_env)
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    nbformat.write(notebook, output / "executed.ipynb")
    report = {"schema": "PASS", "parity": "PASS", "tests": "PASS", "fresh_kernel": "PASS",
              "execution_python": execution_python,
              "worked_cells": sum(c.execution_count is not None for c in code),
              "exercise_cells_skipped": sum("exercise" in c.metadata.get("tags", []) for c in code),
              "skip_messages": [line for cell in code for output in cell.outputs
                                if output.output_type == "stream"
                                for line in output.get("text", "").splitlines() if "SKIP" in line]}
    atomic_json(output / "validation.json", report)
    return report


def agent_command(custom: list[str] | None, worktree: Path, phase: str, final: Path, chapter: dict) -> list[str]:
    values = {"worktree": str(worktree), "phase": phase, "final": str(final), "chapter": chapter["id"]}
    if custom:
        result = []
        for value in custom:
            for key, replacement in values.items():
                value = value.replace("{" + key + "}", replacement)
            result.append(value)
        return result
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex CLI is not on PATH. Use plan now; install/sign in separately before live run.")
    sandbox = "read-only" if phase == "review" else "workspace-write"
    return [executable, "exec", "--sandbox", sandbox, "--json", "-o", str(final), "-"]


class AgentExecutionError(RuntimeError):
    """Provider process failed; pause dispatch instead of starting queued jobs."""


def call_agent(command: list[str], prompt: str, worktree: Path, output: Path, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".jsonl").open("x", encoding="utf-8") as stdout, output.with_suffix(".stderr.log").open("x", encoding="utf-8") as stderr:
        try:
            result = subprocess.run(command, cwd=worktree, input=prompt, text=True, encoding="utf-8",
                                    stdout=stdout, stderr=stderr, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as error:
            raise AgentExecutionError(f"agent timed out after {timeout}s; see {output.with_suffix('.stderr.log')}") from error
        except OSError as error:
            raise AgentExecutionError(f"agent command failed: {error}; see {output.with_suffix('.stderr.log')}") from error
    if result.returncode:
        raise AgentExecutionError(f"agent exited {result.returncode}; see {output.with_suffix('.stderr.log')}")
    if not output.is_file():
        raise AgentExecutionError(f"agent did not create final output: {output}")


def fingerprint(root: Path, manifest: dict, selected: list[dict], request: str) -> str:
    digest = hashlib.sha256(json.dumps([manifest, [c["id"] for c in selected], request], sort_keys=True).encode())
    for relative in manifest.get("shared_context", []):
        digest.update(relative.encode())
        digest.update(inside(root, relative).read_bytes())
    return digest.hexdigest()


def make_plan(root: Path, manifest: dict, selected: list[dict], run_dir: Path, request="") -> dict:
    if run_dir.exists():
        raise ValueError(f"run already exists: {run_dir}; use a new --run-id or run --resume")
    base = git(root, "rev-parse", "HEAD").stdout.strip()
    state = {"version": 1, "base": base, "fingerprint": fingerprint(root, manifest, selected, request),
             "created_at": datetime.now(timezone.utc).isoformat(), "request": request,
             "chapters": {c["id"]: {"status": "planned"} for c in selected}}
    run_dir.mkdir(parents=True)
    for chapter in selected:
        directory = run_dir / chapter["id"]
        directory.mkdir()
        (directory / "author.prompt.md").write_text(author_prompt(manifest, chapter, request), encoding="utf-8")
        (directory / "review.prompt.md").write_text(review_prompt(chapter), encoding="utf-8")
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "state.json", state)
    return state


class SafetyViolation(ValueError):
    """A scope/history violation cannot be retried by the resume command."""


def assert_safe_worktree(worktree: Path, chapter: dict, base: str, branch: str) -> None:
    try:
        check_scope(worktree, chapter)
        if (git(worktree, "rev-parse", "HEAD").stdout.strip() != base
                or git(worktree, "branch", "--show-current").stdout.strip() != branch):
            raise ValueError("agent changed Git history or branch")
    except (ValueError, RuntimeError) as error:
        raise SafetyViolation(str(error)) from error


def snapshot_artifacts(worktree: Path, chapter: dict, destination: Path) -> dict:
    """Keep immutable before/after copies of every owned file for this attempt."""
    destination.mkdir(parents=True, exist_ok=False)
    missing = []
    for relative in chapter["paths"].values():
        source = inside(worktree, relative)
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        else:
            missing.append(relative)
    hashes = artifact_hashes(worktree, chapter)
    atomic_json(destination / "snapshot.json", {"hashes": hashes, "missing": missing})
    return hashes


def run_one(root: Path, chapter: dict, run_dir: Path, state: dict,
            custom: list[str] | None, timeout: int, execution_python: str | None = None,
            resume: bool = False) -> dict:
    identifier = chapter["id"]
    directory = run_dir / identifier
    worktree = run_dir / "worktrees" / identifier
    record = dict(state["chapters"][identifier])
    leaf = directory / "status.json"
    if resume and leaf.is_file():
        record = json.loads(leaf.read_text(encoding="utf-8"))
    if record.get("blocked"):
        raise SafetyViolation(f"{identifier}: blocked safety violation: {record.get('error')}")
    if record["status"] in TERMINAL:
        assert_safe_worktree(worktree, chapter, state["base"], record["branch"])
        if not worktree.is_dir() or artifact_hashes(worktree, chapter) != record.get("hashes"):
            raise SafetyViolation(f"{identifier}: completed artifacts changed; use a new run")
        return record
    branch = f"factory/{run_dir.name}/{identifier}"
    if worktree.exists() and not resume:
        raise ValueError(f"{identifier}: incomplete worktree preserved; use explicit --resume to continue")
    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        git(root, "worktree", "add", "-b", branch, str(worktree), state["base"])
    record.update(worktree=str(worktree), branch=branch)
    record.setdefault("completed", {})
    record.setdefault("attempts", {})
    completed = record["completed"]
    previous_error = record.get("error", "interrupted before a completion checkpoint")
    record.pop("agent_failure", None)
    active_attempt = None
    before = None

    def save():
        atomic_json(leaf, record)

    def begin(phase):
        nonlocal active_attempt
        existing = list((directory / "attempts").glob("[0-9]*-*"))
        last = max((int(p.name.split("-", 1)[0]) for p in existing), default=0)
        record["attempt"] = max(record.get("attempt", 0), last) + 1
        prior_phase = sum(p.name.endswith("-" + phase) for p in existing)
        record["attempts"][phase] = max(record["attempts"].get(phase, 0), prior_phase) + 1
        record.update(stage=phase, status={"author": "authoring", "validation": "validating", "review": "reviewing"}[phase])
        active_attempt = directory / "attempts" / f"{record['attempt']:06d}-{phase}"
        active_attempt.mkdir(parents=True, exist_ok=False)
        save()
        snapshot_artifacts(worktree, chapter, active_attempt / "before")
        return active_attempt

    def finish(error=None):
        nonlocal active_attempt
        if active_attempt is not None:
            attempt = active_attempt
            failure = error
            try:
                snapshot_artifacts(worktree, chapter, attempt / "after")
            except (OSError, ValueError) as snapshot_error:
                failure = error or snapshot_error
                raise
            finally:
                atomic_json(attempt / "result.json", {
                    "stage": record["stage"], "attempt": record["attempt"],
                    "status": "failed" if failure else "completed", "error": str(failure) if failure else None,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })
                active_attempt = None

    try:
        assert_safe_worktree(worktree, chapter, state["base"], branch)
        current = artifact_hashes(worktree, chapter)
        if completed.get("author", {}).get("hashes") != current:
            completed.clear()
            attempt = begin("author")
            prompt = (directory / "author.prompt.md").read_text(encoding="utf-8")
            if resume:
                prompt += ("\nEXPLICIT CONTINUATION: Keep the saved work in the owned files. Read and "
                           "continue those existing artifacts; do not restart or replace them wholesale. "
                           f"Previous failure: {previous_error}\nBefore-attempt snapshots: {attempt / 'before'}\n")
            (attempt / "prompt.md").write_text(prompt, encoding="utf-8")
            final = attempt / "author.final.txt"
            call_agent(agent_command(custom, worktree, "author", final, chapter), prompt, worktree, final, timeout)
            assert_safe_worktree(worktree, chapter, state["base"], branch)
            completed["author"] = {"hashes": artifact_hashes(worktree, chapter), "attempt": record["attempt"]}
            finish()
            save()
        if (completed.get("validation", {}).get("hashes") != artifact_hashes(worktree, chapter)
                or completed.get("validation", {}).get("python") != (execution_python or sys.executable)):
            completed.pop("validation", None)
            completed.pop("review", None)
            attempt = begin("validation")
            build_chapter(worktree, chapter)
            # Notebook regeneration belongs to the completed author's output.
            completed["author"]["hashes"] = artifact_hashes(worktree, chapter)
            save()
            record["validation"] = validate_chapter(worktree, chapter, attempt / "validation", timeout, execution_python)
            assert_safe_worktree(worktree, chapter, state["base"], branch)
            completed["validation"] = {"hashes": artifact_hashes(worktree, chapter),
                                       "python": execution_python or sys.executable, "attempt": record["attempt"]}
            finish()
            save()
        before = artifact_hashes(worktree, chapter)
        if completed.get("review", {}).get("hashes") != before:
            attempt = begin("review")
            prompt = review_prompt(chapter)
            if resume:
                prompt += f"\nResumed review; previous failure: {previous_error}\n"
            (attempt / "prompt.md").write_text(prompt, encoding="utf-8")
            final = attempt / "review.final.json"
            call_agent(agent_command(custom, worktree, "review", final, chapter), prompt, worktree, final, timeout)
            assert_safe_worktree(worktree, chapter, state["base"], branch)
            if artifact_hashes(worktree, chapter) != before:
                raise SafetyViolation("reviewer modified files; read-only review violated")
            review = json.loads(final.read_text(encoding="utf-8"))
            if review.get("approved") is not True or not isinstance(review.get("findings"), list):
                raise ValueError(f"review requires revision: {review}")
            record["review"] = review
            completed["review"] = {"hashes": before, "attempt": record["attempt"]}
            finish()
            save()
        record["stage"] = "candidate"
        save()
        git(worktree, "add", "--", *chapter["paths"].values())
        patch = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"], cwd=worktree, capture_output=True, check=True).stdout
        (directory / "candidate.patch").write_bytes(patch)
        record.update(status="reviewed", hashes=before)
        record.pop("error", None)
        save()
        return record
    except BaseException as error:
        # Persist even on explicit interruption; no retry happens in this call.
        if isinstance(error, AgentExecutionError):
            record["agent_failure"] = True
        try:
            assert_safe_worktree(worktree, chapter, state["base"], branch)
            if record.get("stage") == "review" and before is not None and artifact_hashes(worktree, chapter) != before:
                raise SafetyViolation("reviewer modified files; read-only review violated")
        except SafetyViolation as violation:
            error = violation
        record.update(status="blocked" if isinstance(error, SafetyViolation) else "failed", error=str(error))
        if isinstance(error, SafetyViolation):
            record["blocked"] = True
        try:
            finish(error)
        except (OSError, ValueError) as snapshot_error:
            record["snapshot_error"] = str(snapshot_error)
        save()
        raise error


def integrate(root: Path, run_dir: Path, manifest: dict, state: dict) -> dict:
    """Apply reviewed, disjoint patches to a new candidate worktree only."""
    selected = state["chapters"]
    if any(r["status"] not in TERMINAL for r in selected.values()):
        return {"status": "blocked", "reason": "not every selected chapter passed checks and review"}
    candidate = run_dir / "candidate"
    marker = run_dir / "integration-complete.json"
    expected = {p: digest for record in selected.values() for p, digest in record["hashes"].items()}

    def candidate_matches():
        actual = {p: hashlib.sha256(inside(candidate, p).read_bytes()).hexdigest()
                  for p in expected if inside(candidate, p).is_file()}
        return (actual == expected and changed_paths(candidate) <= expected.keys()
                and git(candidate, "rev-parse", "HEAD").stdout.strip() == state["base"])

    if candidate.exists():
        if marker.is_file():
            completed = json.loads(marker.read_text(encoding="utf-8"))
            if (completed.get("base") == state["base"] and completed.get("hashes") == expected
                    and candidate_matches()):
                return {"status": "preserved", "path": str(candidate),
                        "reason": "completed candidate verified; no files reset"}
        return {"status": "blocked", "path": str(candidate),
                "reason": "partial or modified candidate preserved for inspection; no success is claimed"}
    branch = f"factory/{run_dir.name}/candidate"
    git(root, "worktree", "add", "-b", branch, str(candidate), state["base"])
    try:
        for identifier in selected:
            patch = run_dir / identifier / "candidate.patch"
            if patch.stat().st_size:
                git(candidate, "apply", "--index", str(patch))
        if not candidate_matches():
            raise ValueError("candidate artifacts differ from the reviewed outputs")
    except (OSError, ValueError, RuntimeError) as error:
        return {"status": "blocked", "path": str(candidate), "reason": str(error)}
    atomic_json(marker, {"base": state["base"], "hashes": expected})
    return {"status": "candidate", "path": str(candidate), "branch": branch,
            "note": "Staged review candidate only. No commit, push, merge, or release approval."}


def read_state(run_dir: Path) -> dict:
    """Reconcile finished leaf checkpoints after a coordinator interruption."""
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    for identifier, record in state["chapters"].items():
        progress = run_dir / identifier / "status.json"
        if progress.is_file():
            state["chapters"][identifier] = json.loads(progress.read_text(encoding="utf-8"))
    return state


def resume_plan(root: Path, manifest: dict, selected: list[dict], run_dir: Path, request: str) -> dict:
    state = read_state(run_dir)
    if state["fingerprint"] != fingerprint(root, manifest, selected, request):
        raise ValueError("manifest/context/selection changed; use a new run")
    if state["base"] != git(root, "rev-parse", "HEAD").stdout.strip():
        raise ValueError("source base changed; use a new run")
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "build", "validate", "status"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="config/book_manifest.json")
    parser.add_argument("--chapter", action="append", dest="chapters")
    parser.add_argument("--request", default="")
    parser.add_argument("--downstream", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--python", default=sys.executable, help="Interpreter for reference tests and fresh kernels")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--agent-command", help="Trusted JSON argv; placeholders: worktree, phase, final, chapter")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.workers < 1 or args.workers > 8 or args.timeout < 1:
            raise ValueError("workers must be 1–8 and timeout positive")
        run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if not SAFE_ID.fullmatch(run_id):
            raise ValueError("invalid run ID")
        run_dir = root / ".book-factory" / "runs" / run_id
        if args.command == "status":
            state = read_state(run_dir)
            print(json.dumps(state, indent=2))
            return 0
        manifest = load_manifest(inside(root, args.manifest))
        selected = select_chapters(manifest, args.chapters, args.request, args.downstream)
        if not selected:
            raise ValueError("no managed chapters selected")
        print("Selected chapters:", ", ".join(c["id"] for c in selected), flush=True)
        if args.command == "build":
            for chapter in selected:
                build_chapter(root, chapter)
            return 0
        if args.command == "validate":
            for chapter in selected:
                report = validate_chapter(root, chapter, run_dir / chapter["id"] / "validation", args.timeout, args.python)
                print(chapter["id"], json.dumps(report))
            return 0
        custom = json.loads(args.agent_command) if args.agent_command else None
        if custom is not None and (not isinstance(custom, list) or not custom or any(not isinstance(x, str) for x in custom)):
            raise ValueError("agent-command must be a nonempty JSON string array")
        if args.command == "run":
            # Missing provider is reported before allocating a run or worktrees.
            agent_command(custom, root, "author", run_dir / "probe.txt", selected[0])
            if changed_paths(root):
                raise ValueError("source checkout must be clean and artifacts committed before live run")
        if args.resume:
            if args.command != "run":
                raise ValueError("resume applies only to run")
            state = resume_plan(root, manifest, selected, run_dir, args.request)
        else:
            state = make_plan(root, manifest, selected, run_dir, args.request)
        print("Run:", run_id, "Plan:", run_dir, flush=True)
        if args.command == "plan":
            return 0
        state.pop("dispatch_paused", None)
        remaining = iter(selected)
        paused = False
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}

            def dispatch():
                while not paused and len(futures) < args.workers:
                    chapter = next(remaining, None)
                    if chapter is None:
                        break
                    future = pool.submit(run_one, root, chapter, run_dir, state, custom,
                                         args.timeout, args.python, args.resume)
                    futures[future] = chapter

            dispatch()
            while futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in finished:
                    chapter = futures.pop(future)
                    try:
                        state["chapters"][chapter["id"]] = future.result()
                    except Exception as error:
                        leaf = run_dir / chapter["id"] / "status.json"
                        record = json.loads(leaf.read_text(encoding="utf-8")) if leaf.is_file() else {}
                        record["status"] = "blocked" if isinstance(error, SafetyViolation) else "failed"
                        if isinstance(error, SafetyViolation):
                            record["blocked"] = True
                        record["error"] = str(error)
                        atomic_json(leaf, record)
                        state["chapters"][chapter["id"]] = record
                        if isinstance(error, AgentExecutionError) or record.get("agent_failure"):
                            paused = True
                            state["dispatch_paused"] = {"chapter": chapter["id"], "reason": str(error)}
                    atomic_json(run_dir / "state.json", state)
                    print(chapter["id"], state["chapters"][chapter["id"]]["status"], flush=True)
                dispatch()
        state["integration"] = integrate(root, run_dir, manifest, state)
        atomic_json(run_dir / "state.json", state)
        print(json.dumps(state["integration"], indent=2))
        return 0 if state["integration"]["status"] in {"candidate", "preserved"} else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
