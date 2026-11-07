"""Focused pipeline mechanics; real provider/model usage is not mocked as proof."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import book_factory as factory


def chapter(identifier="03", dependencies=None, managed=True):
    return {
        "id": identifier, "slug": "tensors", "title": "Tensor batches",
        "objectives": ["Inspect shapes"], "topics": ["tensors", "broadcasting"],
        "depends_on": dependencies or [], "tier": "core", "managed": managed,
        "paths": {"manuscript": f"book/draft/{identifier}.md",
                  "notebook": f"book/draft/{identifier}.ipynb",
                  "solutions": f"solutions/draft/ch{identifier}.py",
                  "tests": f"tests/draft/test_ch{identifier}.py"},
    }


def manifest():
    protected = chapter("01", managed=False)
    second = chapter("03", ["01"])
    third = chapter("04", ["03"])
    third.update(title="Model training", slug="training", topics=["training", "optimizer"])
    return {"version": 1, "shared_context": ["BOOK_GOALS.md"], "chapters": [protected, second, third]}


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "C:/secret", "x\\y", ".git/config", "a/../b", "a//b", "a\nfile"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError):
        factory.relative_path(path)


def test_manifest_ownership_and_graph_validation():
    source = manifest()
    assert factory.validate_manifest(source) is source
    duplicate = copy.deepcopy(source)
    duplicate["chapters"][2]["paths"]["tests"] = duplicate["chapters"][1]["paths"]["tests"]
    with pytest.raises(ValueError, match="shared artifact"):
        factory.validate_manifest(duplicate)
    cycle = copy.deepcopy(source)
    cycle["chapters"][1]["depends_on"] = ["04"]
    with pytest.raises(ValueError, match="cycle"):
        factory.validate_manifest(cycle)
    missing = copy.deepcopy(source)
    missing["chapters"][1]["depends_on"] = ["99"]
    with pytest.raises(ValueError, match="unknown dependency"):
        factory.validate_manifest(missing)


@pytest.mark.parametrize("identifier", ["../03", "bad/name", "", "has space", 3])
def test_invalid_ids(identifier):
    source = manifest()
    source["chapters"][1]["id"] = identifier
    with pytest.raises(ValueError, match="chapter ID"):
        factory.validate_manifest(source)


def test_select_protects_existing_and_routes_topics():
    source = manifest()
    assert [c["id"] for c in factory.select_chapters(source)] == ["03", "04"]
    assert [c["id"] for c in factory.select_chapters(source, request="improve broadcasting")] == ["03"]
    assert [c["id"] for c in factory.select_chapters(source, ["03"], downstream=True)] == ["03", "04"]
    with pytest.raises(ValueError, match="protected"):
        factory.select_chapters(source, ["01"])
    with pytest.raises(ValueError, match="unknown chapter"):
        factory.select_chapters(source, ["99"])
    with pytest.raises(ValueError, match="matched no"):
        factory.select_chapters(source, request="octopus")


def test_builder_preserves_prose_tags_and_no_outputs():
    source = "# Lesson\n\nReason.\n\n```python\nx = 2\n```\n\nExercise.\n\n```python exercise\nraise NotImplementedError\n```\n\nCompletion.\n"
    notebook = factory.build_notebook(source, "book/draft/03.md")
    assert notebook == factory.build_notebook(source, "book/draft/03.md")
    assert [c["cell_type"] for c in notebook["cells"]] == ["markdown", "code", "markdown", "code", "markdown"]
    assert notebook["cells"][3]["metadata"]["tags"] == ["exercise"]
    assert notebook["cells"][1]["execution_count"] is None
    assert notebook["cells"][1]["outputs"] == []
    assert "Completion." in "".join(notebook["cells"][-1]["source"])
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))


def test_raw_parity_is_checked_before_source_normalization(tmp_path):
    contract = chapter()
    for role, relative in contract["paths"].items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Stub\n")
    source = "# Lesson\n\n```python\nassert 2 + 2 == 4\n```\n"
    (tmp_path / contract["paths"]["manuscript"]).write_text(source)
    factory.build_chapter(tmp_path, contract)
    notebook = factory.load_chapter_notebook(tmp_path, contract)
    assert notebook.cells[1].source == "assert 2 + 2 == 4\n"
    (tmp_path / contract["paths"]["manuscript"]).write_text(source + "\nAdded explanation.\n")
    with pytest.raises(ValueError, match="differs"):
        factory.load_chapter_notebook(tmp_path, contract)


def init_repo(root):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    (root / "BOOK_GOALS.md").write_text("Focused practical learning\n")
    (root / ".gitignore").write_text(".book-factory/\n__pycache__/\n.pytest_cache/\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base"], cwd=root, check=True)


def test_scope_rejects_untracked_outside_owner(tmp_path):
    init_repo(tmp_path)
    (tmp_path / "unowned.txt").write_text("accident")
    with pytest.raises(ValueError, match="outside chapter"):
        factory.check_scope(tmp_path, chapter())


def test_plan_persists_concrete_ownership_and_context_fingerprint(tmp_path):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data, ["03"])
    directory = tmp_path / ".book-factory/runs/one"
    state = factory.make_plan(tmp_path, data, selected, directory, "add a shape failure")
    prompt = (directory / "03/author.prompt.md").read_text()
    assert "add a shape failure" in prompt
    assert "solutions/draft/ch03.py" in prompt
    assert "Do not commit" in prompt
    assert (directory / "03/review.prompt.md").is_file()
    assert state["chapters"]["03"]["status"] == "planned"
    (tmp_path / "BOOK_GOALS.md").write_text("Changed goal")
    assert factory.fingerprint(tmp_path, data, selected, "add a shape failure") != state["fingerprint"]
    with pytest.raises(ValueError, match="already exists"):
        factory.make_plan(tmp_path, data, selected, directory)


def test_missing_provider_fails_before_allocating_run(tmp_path, monkeypatch):
    init_repo(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr(factory.shutil, "which", lambda name: None)
    assert factory.main(["run", "--root", str(tmp_path), "--manifest", "manifest.json", "--run-id", "missing"]) == 1
    assert not (tmp_path / ".book-factory/runs/missing").exists()


def test_adapter_replaces_only_documented_placeholders(tmp_path):
    command = factory.agent_command(["python", "-c", "print({'ok': True})", "{chapter}", "{final}"],
                                    tmp_path, "review", tmp_path / "final.json", chapter())
    assert command[2] == "print({'ok': True})"
    assert command[3] == "03"
    assert command[4] == str(tmp_path / "final.json")


def test_reviewer_default_sandbox_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(factory.shutil, "which", lambda name: "codex")
    command = factory.agent_command(None, tmp_path, "review", tmp_path / "final", chapter())
    assert command[command.index("--sandbox") + 1] == "read-only"


def fake_agent(tmp_path, scope_violation=False, rejection=False):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "phase, final, chapter = sys.argv[1:]\n"
        "prompt = sys.stdin.read()\n"
        "root = pathlib.Path.cwd()\n"
        "if phase == 'author':\n"
        "    outputs = {f'book/draft/{chapter}.md': '# Lesson\\n\\n```python\\nassert 2 + 2 == 4\\n```\\n', f'solutions/draft/ch{chapter}.py': 'def answer(): return 4\\n', f'tests/draft/test_ch{chapter}.py': 'def test_answer(): assert 2 + 2 == 4\\n'}\n"
        "    for name, content in outputs.items():\n"
        "        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)\n"
        + ("    (root / 'BOOK_GOALS.md').write_text('unauthorized')\n" if scope_violation else "")
        + "    pathlib.Path(final).write_text('authored')\n"
        "else:\n"
        + f"    pathlib.Path(final).write_text(json.dumps({{'approved': {not rejection}, 'findings': []}}))\n"
    )
    return [sys.executable, str(script), "{phase}", "{final}", "{chapter}"]


def test_fake_dispatch_review_candidate_and_resume(tmp_path, monkeypatch):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data, ["03"])
    run = tmp_path / ".book-factory/runs/example"
    state = factory.make_plan(tmp_path, data, selected, run)
    command = fake_agent(tmp_path)
    # This seam tests orchestration only. Real kernels are exercised by the
    # dedicated validation command/CI; this result is deliberately marked fake.
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    record = factory.run_one(tmp_path, selected[0], run, state, command, 30)
    assert record["status"] == "reviewed"
    assert record["validation"] == {"test_double": True}
    state["chapters"]["03"] = record
    result = factory.integrate(tmp_path, run, data, state)
    assert result["status"] == "candidate"
    candidate = Path(result["path"])
    assert (candidate / "book/draft/03.md").is_file()
    assert not (tmp_path / "book/draft/03.md").exists()
    assert factory.run_one(tmp_path, selected[0], run, state, command, 30) == record
    assert factory.integrate(tmp_path, run, data, state)["status"] == "preserved"
    (candidate / "book/draft/03.md").write_text("manual candidate edits")
    assert factory.integrate(tmp_path, run, data, state)["status"] == "blocked"
    Path(record["worktree"], "book/draft/03.md").write_text("edited after review")
    with pytest.raises(ValueError, match="changed"):
        factory.run_one(tmp_path, selected[0], run, state, command, 30)


@pytest.mark.parametrize("violation,rejection", [(True, False), (False, True)])
def test_fake_failure_never_integrates(tmp_path, monkeypatch, violation, rejection):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data, ["03"])
    run = tmp_path / ".book-factory/runs/failure"
    state = factory.make_plan(tmp_path, data, selected, run)
    command = fake_agent(tmp_path, violation, rejection)
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    with pytest.raises(ValueError):
        factory.run_one(tmp_path, selected[0], run, state, command, 30)
    state["chapters"]["03"] = {"status": "failed"}
    assert factory.integrate(tmp_path, run, data, state)["status"] == "blocked"
    assert not (run / "candidate").exists()


def test_validation_failure_propagates_before_reviewer(tmp_path, monkeypatch):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data, ["03"])
    run = tmp_path / ".book-factory/runs/validation-failure"
    state = factory.make_plan(tmp_path, data, selected, run)
    command = fake_agent(tmp_path)

    def fail(*args):
        raise RuntimeError("reference tests failed")

    monkeypatch.setattr(factory, "validate_chapter", fail)
    with pytest.raises(RuntimeError, match="reference tests failed"):
        factory.run_one(tmp_path, selected[0], run, state, command, 30)
    assert not (run / "03/review.final.json").exists()


def test_incomplete_worktree_is_preserved(tmp_path):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data, ["03"])
    run = tmp_path / ".book-factory/runs/interrupted"
    state = factory.make_plan(tmp_path, data, selected, run)
    worktree = run / "worktrees/03"
    worktree.mkdir(parents=True)
    (worktree / "partial.txt").write_text("valuable partial work")
    with pytest.raises(ValueError, match="preserved"):
        factory.run_one(tmp_path, selected[0], run, state, ["unused"], 10)
    assert (worktree / "partial.txt").read_text() == "valuable partial work"


def test_partial_integration_never_becomes_success_on_resume(tmp_path, monkeypatch):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data)
    run = tmp_path / ".book-factory/runs/partial"
    state = factory.make_plan(tmp_path, data, selected, run)
    command = fake_agent(tmp_path)
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    for contract in selected:
        state["chapters"][contract["id"]] = factory.run_one(tmp_path, contract, run, state, command, 30)
    (run / "04/candidate.patch").write_text("not a valid patch")
    first = factory.integrate(tmp_path, run, data, state)
    assert first["status"] == "blocked"
    assert (run / "candidate/book/draft/03.md").is_file()
    assert not (run / "candidate/book/draft/04.md").exists()
    assert not (run / "integration-complete.json").exists()
    assert factory.integrate(tmp_path, run, data, state)["status"] == "blocked"


def test_leaf_completion_is_reconciled_after_interruption(tmp_path):
    run = tmp_path / "run"
    factory.atomic_json(run / "state.json", {"chapters": {"03": {"status": "planned"}}})
    factory.atomic_json(run / "03/status.json", {"status": "reviewed", "hashes": {"example": "digest"}})
    assert factory.read_state(run)["chapters"]["03"]["status"] == "reviewed"


def resumable_fake_agent(root, *, author_quota=False, review_quota=False):
    """A process-level adapter that saves a real partial edit before quota exit."""
    script = root / "resumable_fake.py"
    script.write_text(
        "import json,pathlib,sys\n"
        "phase,final,chapter=sys.argv[1:]\n"
        "prompt=sys.stdin.read()\n"
        "work=pathlib.Path.cwd(); ledger=pathlib.Path(final).parents[2]/'fake-counts.json'\n"
        "counts=json.loads(ledger.read_text()) if ledger.exists() else {}\n"
        "counts[phase]=counts.get(phase,0)+1; ledger.write_text(json.dumps(counts))\n"
        "manuscript=work/f'book/draft/{chapter}.md'\n"
        "if phase=='author':\n"
        "    manuscript.parent.mkdir(parents=True,exist_ok=True)\n"
        + ("    if counts[phase]==1 and chapter=='03':\n"
           "        manuscript.write_text('# Saved partial draft\\n\\nKeep this paragraph.\\n')\n"
           "        print('quota exceeded',file=sys.stderr); sys.exit(75)\n"
           "    if chapter=='03':\n"
           "        assert 'EXPLICIT CONTINUATION' in prompt\n"
           "        assert 'quota' not in manuscript.read_text()\n"
           "        assert 'Keep this paragraph.' in manuscript.read_text()\n" if author_quota else "")
        + "    existing=manuscript.read_text() if manuscript.exists() else '# Lesson\\n'\n"
        "    manuscript.write_text(existing+'\\n```python\\nassert 2+2==4\\n```\\n')\n"
        "    for relative,text in [(f'solutions/draft/ch{chapter}.py','def answer(): return 4\\n'),(f'tests/draft/test_ch{chapter}.py','def test_answer(): assert 2+2==4\\n')]:\n"
        "        target=work/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(text)\n"
        "    pathlib.Path(final).write_text('author complete')\n"
        "else:\n"
        + ("    if counts[phase]==1:\n"
           "        print('quota exceeded',file=sys.stderr); sys.exit(75)\n" if review_quota else "")
        + "    pathlib.Path(final).write_text(json.dumps({'approved':True,'findings':[],'summary':'fake review'}))\n"
    )
    return [sys.executable, str(script), "{phase}", "{final}", "{chapter}"]


def setup_resume_run(tmp_path, run_id):
    init_repo(tmp_path)
    data = manifest()
    selected = factory.select_chapters(data, ["03"])
    run = tmp_path / ".book-factory/runs" / run_id
    state = factory.make_plan(tmp_path, data, selected, run)
    return data, selected[0], run, state


def test_quota_resume_keeps_partial_edits_snapshots_and_old_logs(tmp_path, monkeypatch):
    data, contract, run, state = setup_resume_run(tmp_path, "quota")
    command = resumable_fake_agent(tmp_path, author_quota=True)
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    with pytest.raises(RuntimeError, match="agent exited 75"):
        factory.run_one(tmp_path, contract, run, state, command, 30)
    leaf = json.loads((run / "03/status.json").read_text())
    assert leaf["status"] == "failed"
    assert leaf["stage"] == "author"
    assert leaf["attempts"] == {"author": 1}
    old_stderr = run / "03/attempts/000001-author/author.final.stderr.log"
    old_bytes = old_stderr.read_bytes()
    saved = run / "03/attempts/000001-author/after/book/draft/03.md"
    assert "Keep this paragraph." in saved.read_text()
    restored = factory.resume_plan(tmp_path, data, [contract], run, "")
    result = factory.run_one(tmp_path, contract, run, restored, command, 30, resume=True)
    assert result["status"] == "reviewed"
    assert result["attempts"] == {"author": 2, "validation": 1, "review": 1}
    assert old_stderr.read_bytes() == old_bytes
    assert json.loads((old_stderr.parent / "result.json").read_text())["status"] == "failed"
    assert "agent exited 75" in json.loads((old_stderr.parent / "result.json").read_text())["error"]
    assert (run / "03/attempts/000002-author/before/book/draft/03.md").read_bytes() == saved.read_bytes()
    assert "Keep this paragraph." in Path(result["worktree"], "book/draft/03.md").read_text()


def test_validation_resume_does_not_repeat_successful_author(tmp_path, monkeypatch):
    data, contract, run, state = setup_resume_run(tmp_path, "validation-retry")
    command = resumable_fake_agent(tmp_path)
    calls = []

    def validation(*args):
        calls.append(args[2])
        if len(calls) == 1:
            raise RuntimeError("environment temporarily unavailable")
        return {"test_double": True}

    monkeypatch.setattr(factory, "validate_chapter", validation)
    with pytest.raises(RuntimeError, match="temporarily"):
        factory.run_one(tmp_path, contract, run, state, command, 30)
    resumed = factory.resume_plan(tmp_path, data, [contract], run, "")
    result = factory.run_one(tmp_path, contract, run, resumed, command, 30, resume=True)
    assert result["attempts"] == {"author": 1, "validation": 2, "review": 1}
    assert len(set(calls)) == 2
    assert json.loads((run / "03/fake-counts.json").read_text()) == {"author": 1, "review": 1}


def test_review_quota_resume_skips_author_and_validation(tmp_path, monkeypatch):
    data, contract, run, state = setup_resume_run(tmp_path, "review-retry")
    command = resumable_fake_agent(tmp_path, review_quota=True)
    validations = []
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: validations.append(True) or {"test_double": True})
    with pytest.raises(RuntimeError, match="agent exited 75"):
        factory.run_one(tmp_path, contract, run, state, command, 30)
    resumed = factory.resume_plan(tmp_path, data, [contract], run, "")
    result = factory.run_one(tmp_path, contract, run, resumed, command, 30, resume=True)
    assert result["attempts"] == {"author": 1, "validation": 1, "review": 2}
    assert len(validations) == 1
    assert len(list((run / "03/attempts").glob("*-review/review.final.stderr.log"))) == 2


def test_resume_context_change_is_blocked_before_agent(tmp_path):
    data, contract, run, state = setup_resume_run(tmp_path, "context")
    (tmp_path / "BOOK_GOALS.md").write_text("Different goals")
    with pytest.raises(ValueError, match="context"):
        factory.resume_plan(tmp_path, data, [contract], run, "")
    assert not (run / "worktrees").exists()


def test_scope_violation_stays_blocked_on_explicit_resume(tmp_path, monkeypatch):
    data, contract, run, state = setup_resume_run(tmp_path, "sticky-scope")
    command = fake_agent(tmp_path, scope_violation=True)
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    with pytest.raises(factory.SafetyViolation):
        factory.run_one(tmp_path, contract, run, state, command, 30)
    leaf = json.loads((run / "03/status.json").read_text())
    assert leaf["blocked"] is True
    assert leaf["attempts"]["author"] == 1
    resumed = factory.resume_plan(tmp_path, data, [contract], run, "")
    with pytest.raises(factory.SafetyViolation, match="blocked safety"):
        factory.run_one(tmp_path, contract, run, resumed, command, 30, resume=True)
    assert json.loads((run / "03/status.json").read_text())["attempts"]["author"] == 1


def test_agent_quota_pauses_unstarted_chapters_and_explicit_resume_completes(tmp_path, monkeypatch):
    init_repo(tmp_path)
    command = resumable_fake_agent(tmp_path, author_quota=True)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()))
    factory.git(tmp_path, "add", ".")
    factory.git(tmp_path, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "manifest and fake adapter")
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    args = ["run", "--root", str(tmp_path), "--manifest", "manifest.json", "--run-id", "bounded",
            "--workers", "1", "--agent-command", json.dumps(command)]
    assert factory.main(args) == 1
    run = tmp_path / ".book-factory/runs/bounded"
    failed = factory.read_state(run)
    assert failed["chapters"]["03"]["status"] == "failed"
    assert failed["chapters"]["04"]["status"] == "planned"
    assert failed["dispatch_paused"]["chapter"] == "03"
    assert not (run / "worktrees/04").exists()
    assert factory.main([*args, "--resume"]) == 0
    completed = factory.read_state(run)
    assert all(record["status"] == "reviewed" for record in completed["chapters"].values())
    assert "dispatch_paused" not in completed
    assert completed["integration"]["status"] == "candidate"


def test_candidate_export_resume_does_not_repeat_successful_review(tmp_path, monkeypatch):
    data, contract, run, state = setup_resume_run(tmp_path, "export-retry")
    command = resumable_fake_agent(tmp_path)
    monkeypatch.setattr(factory, "validate_chapter", lambda *args: {"test_double": True})
    original_git = factory.git
    export_attempts = []

    def flaky_git(root, *arguments, **kwargs):
        if arguments[0] == "add" and root.name == "03":
            export_attempts.append(True)
            if len(export_attempts) == 1:
                raise RuntimeError("temporary export failure")
        return original_git(root, *arguments, **kwargs)

    monkeypatch.setattr(factory, "git", flaky_git)
    with pytest.raises(RuntimeError, match="export failure"):
        factory.run_one(tmp_path, contract, run, state, command, 30)
    resumed = factory.resume_plan(tmp_path, data, [contract], run, "")
    result = factory.run_one(tmp_path, contract, run, resumed, command, 30, resume=True)
    assert result["attempts"] == {"author": 1, "validation": 1, "review": 1}
    assert json.loads((run / "03/fake-counts.json").read_text()) == {"author": 1, "review": 1}
