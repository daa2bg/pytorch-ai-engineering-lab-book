# Run the book-production pipeline

This is a small Python orchestration project you can study alongside the book.
The manifest describes chapter ownership and prerequisites. The runner turns a
change request into concrete author/reviewer jobs, isolates edits with Git
worktrees, validates each result, and assembles a review candidate. It never
pushes, merges, or publishes a release.

For the code and persistence walkthrough, read [How the factory works](BOOK_FACTORY_INTERNALS.md).

## First run: plan without model usage

Use Python 3.11+, Git, and a checkout containing the committed factory files.
The plan command needs only the standard library:

```powershell
python tools/book_factory.py plan --run-id first-outline
python tools/book_factory.py plan --chapter 03 --run-id first-chapter
python tools/book_factory.py plan --request "improve tensor broadcasting" --run-id tensors-v2
```

The command prints the selected chapter IDs and writes exact author/reviewer
prompts plus the manifest snapshot and state to
`.book-factory/runs/<run-id>/`. Inspect that plan before spending agent usage.
Routing is deterministic keyword matching against titles, slugs, and topics,
not an LLM pretending to understand every request. Use `--chapter` (repeatable)
when precision matters. No match is an error rather than a whole-book rewrite.
`--downstream` also selects dependent managed chapters for a consistency pass.
Existing reference chapters marked `managed: false` are protected from authoring.

Dependencies describe the reader's learning prerequisites. Draft authors are
independent and can run concurrently; they must not import unpublished code from
another author's chapter. Integration and editorial review handle book-wide
continuity after the first drafts exist.

## Run agents locally

Install/sign in to Codex CLI separately using the official non-interactive-mode
documentation: https://developers.openai.com/codex/noninteractive . The runner
does not install a CLI, purchase usage, or configure credentials. Existing ChatGPT
or API authentication must already work for your chosen CLI; quotas still apply.

Install the project's validation dependencies in your environment, then run:

```powershell
python -m pip install -e ".[dev,book]"
codex --version
python tools/book_factory.py run --chapter 03 --run-id first-live --workers 1
python tools/book_factory.py status --run-id first-live
```

The `book` extra supplies the data, Hugging Face, and serving packages used across
the draft. PyTorch is a base dependency; choose its CPU or accelerator build for
your machine before installing the project. Downloaded model weights are separate
and remain explicitly opt-in in the relevant studies.

Default concurrency is three, capped at eight. Start with one chapter/worker to
learn usage and failure behavior. Each author or reviewer has a bounded wall-clock
timeout (`--timeout`, default 1200 seconds). An agent can still spend usage before
the timeout; the runner is not a token/cost accounting service.

Only up to `--workers` chapter jobs are submitted at once. If an agent process exits
nonzero, times out, cannot launch, or omits its final response, dispatch pauses:
chapters not yet started remain planned and `dispatch_paused` records the cause.
In-flight chapter jobs may finish. This fail-closed rule does not depend on recognizing
a provider's exact quota wording. Fix/reset the account or command and explicitly
resume the same run; the pipeline never polls quotas or retries automatically.
An ordinary chapter validation failure is recorded for that chapter without being
misclassified as an account-wide provider failure.

For each selected chapter the pipeline:

1. Creates a new `factory/<run-id>/<chapter-id>` Git branch/worktree at the source
   commit. The source checkout must be clean; no existing worktree is reset.
2. Calls `codex exec --sandbox workspace-write --json -o <final-file> -`, sending
   the scoped prompt over stdin and recording stdout/stderr separately.
3. Rejects edits outside the four manifest-owned artifacts or changed Git history.
4. Regenerates the complete notebook from the Markdown manuscript, then checks
   notebook schema/parity, runs the chapter's reference pytest tests, and executes
   worked cells in a fresh Jupyter kernel. Tagged exercises remain unfinished.
5. Calls a separate reviewer with `--sandbox read-only`, requiring a JSON decision. Rejects review
   edits and any result not explicitly approved.
6. Only if every selected chapter passes, applies disjoint patches into a new
   `factory/<run-id>/candidate` worktree. Changes are staged for your review;
   the source checkout and base branch are not modified.

Candidate completion is recorded only after all patches apply and its artifact
hashes match the reviewed chapter outputs. An interrupted/partial integration or
a manually changed candidate stays preserved but blocked on resume, never reported
as a successful candidate merely because its directory exists.

Per-chapter checks do not prove whole-book release readiness. Inspect the candidate,
run the whole suite, review chapter transitions and sources, and use the release
plan before manually committing/pushing or merging anything. Path checks detect
violations after execution; they are not a security sandbox. Run only trusted
agents/notebooks, with the CLI's normal sandbox enabled.

## Build, validate, resume

```powershell
python tools/book_factory.py build --chapter 03
python tools/book_factory.py validate --chapter 03 --run-id local-check
python tools/book_factory.py validate --chapter 03 --python C:\Path\To\MLEnv\python.exe --run-id ml-check
python tools/book_factory.py run --chapter 03 --run-id first-chapter --resume
```

`build` uses only Python's standard library. Every `python` fence becomes worked
code; `python exercise` fences gain the `exercise` tag. All other prose remains in
the complete notebook. No reference answers are copied into learner artifacts.

`validate` needs pytest, nbformat, nbclient, and ipykernel plus that chapter's
dependencies. Missing packages, failing tests, and kernel failures fail the run.
The executed notebook, pytest log, and JSON validation report are separate from
the unexecuted learner notebook. The factory does not silently install dependencies
or turn failures into exercise tags.

Validation reports also retain `SKIP` messages printed by executed worked cells,
so optional downloads/provider features are visible rather than implied tested.

`--python` selects the interpreter used for chapter tests and the fresh notebook
kernel. The control interpreter running this script must provide nbformat,
nbclient, and jupyter-client; the selected ML interpreter needs pytest, ipykernel,
and that chapter's packages. This lets an existing ML environment supply PyTorch
without duplicating the control environment's notebook tooling.

`--resume` requires the same run ID, source commit, manifest, context, selection,
and request. It is an explicit recovery action, not an automatic retry loop:

```powershell
# After a quota reset or fixing the validation environment, use the SAME inputs.
python tools/book_factory.py run --chapter 03 --run-id first-live --resume --workers 1
```

Saved author edits remain in their worktree. The resumed author receives a
continuation prompt telling it to read and preserve existing owned artifacts,
with the previous failure and a before-attempt snapshot path. Successful author,
validation, and review checkpoints are skipped while their artifact hashes agree;
a validation environment failure therefore does not restart a successful author,
and a reviewer quota failure does not repeat successful tests. Changing the test/
kernel interpreter invalidates the validation checkpoint. Changed owned files
invalidate the downstream checks; any necessary author invocation is a continuation,
not a forced regeneration from an empty directory. A reviewed chapter changed after
completion remains blocked rather than silently accepted.

Every explicit attempt has a unique `attempts/000001-author/` (or validation/review)
directory containing the prompt where applicable, before/after copies of all owned
artifacts (including a list of missing ones), final response, stdout/stderr, and
`result.json` with the outcome. Status retains the active stage and attempt counters
even on errors. Later attempts do not overwrite prior logs or snapshots. Finished
leaf checkpoints are reconciled after a coordinator interruption. Snapshots preserve
the exact saved files; they cannot guarantee that an agent will make a good revision,
so review the resulting diff.

Scope violations, changed Git history/branches, and reviewer edits are sticky safety
blocks. `--resume` checks them before another provider call and does not clear them
even if a later edit hides the symptom. Inspect/salvage such work and start a new run.
Partial or modified integration candidates also remain blocked. No files/worktrees
are deleted and no automatic quota polling, retry loop, push, or merge is performed.

## Adapter and tests

`--agent-command` accepts a trusted JSON argv array with `{worktree}`, `{phase}`,
`{final}`, and `{chapter}` placeholders. The command reads the prompt on stdin,
writes its final response to `{final}`, and exits nonzero on failure. Author final
output can be prose; reviewer final output must be JSON with `approved`, `findings`,
and `summary`. The runner uses argument lists with `shell=False`; do not embed
shell pipelines in the array. This seam supports other CLIs and fake-agent tests.

```powershell
python -m pytest tests/test_book_factory.py -q
```

The fake-adapter tests cover dispatch, scope rejection, review rejection, validation
failure propagation, candidate isolation, and completed-artifact resume. They mock
the expensive kernel validation seam explicitly; they do not establish that real
Codex dispatch or a real kernel worked on your machine. Run a live pilot separately.

First drafts remain drafts. The workflow makes repeated enrichment reviewable;
it does not make generated explanations authoritative or publication-ready.
