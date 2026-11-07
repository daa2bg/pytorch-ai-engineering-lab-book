"""Assemble the full draft without changing any source chapter.

Usage: python tools/assemble_book.py
Managed chapters require manuscripts; protected references may use notebooks.
The generated book remains a draft even if all source files are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_markdown(notebook: dict) -> str:
    """Preserve prose and code, never execution outputs, from a reference."""
    blocks = []
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        source = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] == "markdown":
            blocks.append(source.strip())
        elif cell["cell_type"] == "code":
            label = "python exercise" if "exercise" in cell.get("metadata", {}).get("tags", []) else "python"
            blocks.append(f"```{label}\n{source.strip()}\n```")
    return "\n\n".join(blocks)


def assemble(root: Path, manifest: dict) -> tuple[str, list[str]]:
    sections = [
        "# PyTorch & AI Engineering Lab — First Draft",
        "A practical path from experienced software developer to ML/AI engineering. "
        "This is a working first draft for study and refinement, not a release candidate. "
        "Run each chapter's own notebook; solutions and tests are separate. "
        "See VALIDATION.md for actual execution evidence and skipped integrations.",
        "## Reading map\n\n" + "\n".join(
            f"- {c['id']}. {c['title']} ({c['tier']})" for c in manifest["chapters"]
        ),
    ]
    missing = []
    for chapter in manifest["chapters"]:
        manuscript = root / chapter["paths"]["manuscript"]
        notebook = root / chapter["paths"]["notebook"]
        if manuscript.is_file():
            body = manuscript.read_text(encoding="utf-8")
        elif not chapter.get("managed", True) and notebook.is_file():
            body = notebook_markdown(json.loads(notebook.read_text(encoding="utf-8")))
        else:
            missing.append(chapter["id"])
            continue
        note = "Protected existing reference" if not chapter.get("managed", True) else "New first-draft study"
        sections.extend([
            f"---\n\n## Study {chapter['id']} — {chapter['title']}\n\n"
            f"{note}. Source notebook: `{chapter['paths']['notebook']}`.",
            body.strip(),
        ])
    return "\n\n".join(sections) + "\n", missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/book_manifest.json")
    parser.add_argument("--output", default="book/draft/BOOK_DRAFT.md")
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    content, missing = assemble(ROOT, manifest)
    if missing:
        print("INCOMPLETE: missing manuscripts for " + ", ".join(missing))
        return 1
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"Assembled {len(manifest['chapters'])} chapters into {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
