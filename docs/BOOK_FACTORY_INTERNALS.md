# How the book factory works

The factory is a small Python job runner around an agent CLI. It uses Git worktrees
for isolated files, JSON for checkpoints, subprocesses for agents and tests, and
a bounded thread pool for concurrency. There is no queue service or database to
operate. The source is `tools/book_factory.py`; the operating commands are in
[BOOK_FACTORY.md](BOOK_FACTORY.md).

## How this draft was created

We first fixed the book goals and outlined small chapter contracts. Chat agents
then wrote different chapters while another agent implemented the reusable local
runner. Integration rebuilt notebooks, tested answers, reviewed drafts, and ran
fresh-kernel validation in GitHub Actions. These chat agents were not launched
through the new local CLI runner. Its orchestration was tested with fake agent
processes; real local Codex dispatch still needs an installed, authenticated CLI
and a small live pilot.

## Follow one chapter through the code

| Component | Function / file | Responsibility |
| --- | --- | --- |
| Task contracts | `config/book_manifest.json` | Objectives, prerequisites, topics, primary sources, and four owned artifact paths |
| Selection | `select_chapters()` | Explicit IDs or keyword matches; optional downstream expansion |
| Planning | `make_plan()`, `author_prompt()` | Save prompts, manifest snapshot, source commit, requirements fingerprint, and planned statuses |
| Worker | `run_one()` | Run one chapter through author, validation, reviewer, and patch export |
| Provider adapter | `agent_command()`, `call_agent()` | Start a subprocess, supply prompt on stdin, capture output, enforce timeout |
| Validation | `build_chapter()`, `validate_chapter()` | Generate notebook, check parity/schema, run pytest, execute worked cells in a fresh kernel |
| Recovery | `resume_plan()`, `read_state()` | Verify the original inputs and recover the latest saved chapter checkpoints |
| Integration | `integrate()` | Apply all approved chapter patches to a separate staged candidate |

The manifest is data, not an intelligent planner. A phrase such as "tensor
broadcasting" matches chapter topics. Ambiguous or unmatched requests need explicit
chapter IDs. Adding a new study means first updating the outline and manifest,
choosing unique artifact paths, and committing those shared requirements. The
chapter author is not authorized to rewrite the curriculum itself.

A Git worktree is another checkout sharing the repository's object/history store.
Each author receives a branch such as `factory/tensors-v2/04` and its own directory.
Two authors can therefore edit their own files without sharing notebook state or
overwriting each other's working copies. The runner checks changed paths against
the chapter's four-file ownership contract.

The author and reviewer are separate agent invocations. They may use the same
configured model, but receive different prompts and permissions: author writes
the owned files; reviewer is read-only and returns a JSON decision. The worker
accepts explicit approval, not prose that merely sounds positive.

## Why the Python process can run several agents

`main()` uses `ThreadPoolExecutor` with at most `--workers` chapter jobs active.
The threads mostly wait for external agent processes and validation processes.
This is local concurrency, not a distributed scheduler. Inside each chapter,
authoring precedes validation, which precedes review. The manifest's prerequisite
graph describes learning order; it does not serialize every authoring task.

The default adapter invokes `codex exec` with a prompt through stdin and distinct
final-response, stdout, and stderr files. `--agent-command` provides another trusted
argv-based adapter. The CLI handles model access and authentication; the runner
does not contain an API key, install models, or bypass account quotas.

## What survives a quota interruption

Run data lives under `.book-factory/runs/<run-id>/`:

- `state.json`: source identity, selected chapters, coordinator status, pause reason.
- `manifest.json`: the contracts used for this run.
- `<chapter>/status.json`: current stage, attempts, hashes, completed checkpoints.
- `<chapter>/attempts/<number>-<stage>/`: prompt, before/after artifact snapshots,
  logs, final response, and an outcome record where the attempt reaches cleanup.
- `worktrees/<chapter>/`: saved partial files the next author invocation continues.
- `candidate/`: the integrated review checkout, created only after all jobs pass.

JSON state is written to a temporary sibling and then replaced, reducing the risk
of reading half a JSON document. Before restarting, hashes verify that previously
approved files still match. Hashes identify saved bytes; they do not establish
editorial quality. Attempt directories keep old logs and snapshots separate.

For example, suppose authoring and validation succeed but the reviewer hits a
quota. The failed review is recorded; unstarted chapters remain planned. Existing
in-flight jobs may finish. After the account is usable, an explicit `--resume`
rechecks source/context and file hashes, then retries review without paying for
authoring again. If authoring itself stopped, the next invocation reads the saved
files and receives a continuation prompt. This continues artifacts, not a restored
model's private conversational state.

There is no automatic quota polling or retry loop. Changed requirements need a new
run. Scope/history violations and modified integration candidates remain blocked
for inspection. A hard stop can lose unsaved agent output; saved files and already
written checkpoints remain. The run directory is Git-ignored and local: GitHub
backups of committed manuscripts do not back up these runtime checkpoints. Keep
the checkout and run directory together if you need to move or back up a run.
Use one coordinator process per run ID; concurrent resumes of the same run are
not a supported locking/distributed-execution mode.

## Operate and inspect it

From a clean committed checkout, after installing/authenticating the CLI:

```powershell
python tools/book_factory.py run --chapter 03 --run-id ch03-refine --workers 1
python tools/book_factory.py status --run-id ch03-refine
# Following an interruption, repeat the same selection/request and run ID:
python tools/book_factory.py run --chapter 03 --run-id ch03-refine --workers 1 --resume
```

For a no-provider exercise, use `plan --chapter 03 --run-id inspect-ch03`, inspect
its author/reviewer prompts and `state.json`, then read the quota-resume tests in
`tests/test_book_factory.py`. Those tests deliberately write a partial draft,
fail the fake agent, and prove that continuation keeps the paragraph and earlier
logs. They also verify that a failed reviewer does not rerun a successful author.

The last stage produces staged changes in a candidate worktree. It does not commit,
push, merge, or claim publication readiness. Review that diff and whole-book
continuity before promoting it. This separates a process completing successfully
from a chapter actually being good enough to publish.
