"""Explicit, bounded acquisition of the book's three public pretrained assets.

This module imports no ML frameworks. Local overrides are caller-provided assets;
prepared snapshots additionally carry verified byte hashes and Hub provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import quote

MAX_BYTES = 600_000_000  # decimal MB, aggregate selected model payloads
SPECS = {
    "bert": ("prajjwal1/bert-tiny", "AI_LAB_MODEL_DIR", "mit"),
    "encoder": ("sentence-transformers/all-MiniLM-L6-v2", "AI_LAB_ENCODER_DIR", "apache-2.0"),
    "generator": ("google/flan-t5-small", "AI_LAB_GENERATOR_DIR", "apache-2.0"),
}
SUPPORT_FILES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.txt", "spiece.model",
    "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt",
}
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def cache_root(cache_dir=None) -> Path:
    return Path(cache_dir or os.environ.get("AI_LAB_CACHE_DIR") or
                Path(__file__).resolve().parents[3] / ".cache" / "ai_lab").resolve()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _matches_upstream(path: Path, size: int, digest: dict) -> bool:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        return False
    if digest["algorithm"] == "sha256":
        return _hash(path) == digest["value"]
    hasher = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest["value"]


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_local(path: Path) -> Path:
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"Incomplete local model directory: {path}")
    if not any((path / name).is_file() for name in WEIGHTS):
        raise ValueError(f"No supported single-file PyTorch weights in {path}")
    if not any((path / name).is_file() for name in ("vocab.txt", "tokenizer.json", "spiece.model")):
        raise ValueError(f"No tokenizer vocabulary in {path}")
    return path


def asset_metadata(path: str | Path) -> dict:
    """Read and verify a prepared snapshot; raise on any missing/corrupted bytes."""
    path = _validate_local(Path(path).resolve())
    data = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", data.get("revision", "")):
        raise ValueError("Invalid asset metadata revision/schema")
    files = data.get("files", {})
    if not isinstance(files, dict) or "config.json" not in files or sum(n in files for n in WEIGHTS) != 1:
        raise ValueError("Invalid asset file manifest")
    total = 0
    for name, details in files.items():
        if name not in SUPPORT_FILES | set(WEIGHTS):
            raise ValueError("Unexpected asset filename")
        target = path / name
        if target.is_symlink() or not target.is_file() or target.stat().st_size != details["size"] or _hash(target) != details["sha256"]:
            raise ValueError(f"Asset integrity failure: {name}")
        total += target.stat().st_size
    if total != data.get("actual_bytes") or total > MAX_BYTES:
        raise ValueError("Asset byte accounting mismatch")
    return data


def _cached(kind: str, root: Path) -> Path | None:
    pointer = root / "models" / kind / "current.json"
    if not pointer.exists():
        return None
    revision = json.loads(pointer.read_text(encoding="utf-8")).get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid cached model revision")
    path = pointer.parent / revision
    metadata = asset_metadata(path)
    if metadata["repo_id"] != SPECS[kind][0] or metadata["revision"] != revision:
        raise ValueError("Cached model identity mismatch")
    return path


def resolve_asset(kind: str, *, cache_dir=None, allow_downloads=None) -> Path:
    """Return a complete local snapshot; network requires explicit opt-in.

    Framework calls must use this path with local_files_only=True and
    trust_remote_code=False. An explicit local override takes precedence and is
    never silently replaced. Its provenance remains the caller's responsibility.
    """
    if kind not in SPECS:
        raise ValueError(f"Unknown asset kind: {kind}")
    override = os.environ.get(SPECS[kind][1])
    if override:
        path = _validate_local(Path(override).expanduser().resolve())
        if (path / "metadata.json").exists():
            metadata = asset_metadata(path)
            if metadata["repo_id"] != SPECS[kind][0]:
                raise ValueError("Local asset identity mismatch")
        return path
    root = cache_root(cache_dir)
    cached = _cached(kind, root)
    if cached is not None:
        return cached
    allowed = os.environ.get("AI_LAB_ALLOW_DOWNLOADS") == "1" if allow_downloads is None else allow_downloads is True
    if not allowed:
        raise FileNotFoundError(f"Missing {kind} asset. Run python tools/prepare_book_assets.py --allow-downloads, or set {SPECS[kind][1]} to a complete local snapshot.")
    return prepare_assets((kind,), cache_dir=root, allow_downloads=True)[kind]


def _model_info(repo_id: str):
    from huggingface_hub import HfApi
    return HfApi(token=False).model_info(repo_id, files_metadata=True, timeout=20)


def _plan(kind: str) -> dict:
    repo_id, _, license_id = SPECS[kind]
    info = _model_info(repo_id)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
        raise ValueError("Hub did not return an immutable commit SHA")
    available = {entry.rfilename: entry for entry in info.siblings}
    weight = next((name for name in WEIGHTS if name in available), None)
    if weight is None or "config.json" not in available or "README.md" not in available:
        raise ValueError(f"Required weights/config/model card missing for {repo_id}")
    names = sorted((SUPPORT_FILES & available.keys()) | {weight})
    if not {"vocab.txt", "tokenizer.json", "spiece.model"}.intersection(names):
        raise ValueError("Model has no supported tokenizer")
    sizes = {name: available[name].size for name in names}
    if any(not isinstance(size, int) or size < 0 for size in sizes.values()):
        raise ValueError("Hub file size missing; refusing unbounded download")
    upstream_hashes = {}
    for name in names:
        entry = available[name]
        lfs = getattr(entry, "lfs", None)
        value = getattr(lfs, "sha256", None) if lfs is not None else None
        algorithm = "sha256" if value else "git-sha1"
        value = value or getattr(entry, "blob_id", None)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}" if algorithm == "sha256" else r"[0-9a-f]{40}", value):
            raise ValueError("Hub integrity digest missing; refusing unverifiable download")
        upstream_hashes[name] = {"algorithm": algorithm, "value": value}
    return {"schema_version": 1, "repo_id": repo_id, "revision": info.sha,
            "source": f"https://huggingface.co/{repo_id}/tree/{info.sha}",
            "license": license_id, "license_source": "repository model card; see saved README.md",
            "retrieved_at": datetime.now(timezone.utc).isoformat(), "planned_bytes": sum(sizes.values()),
            "sizes": sizes, "upstream_hashes": upstream_hashes,
            "weight_format": weight, "trust_remote_code": False}


def _download(url: str, target: Path, expected: int) -> None:
    """Fail closed if actual bytes exceed metadata or a file takes over 300 s."""
    import requests
    start = time.monotonic()
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=(20, 20)) as response:
            response.raise_for_status()
            count = 0
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    count += len(chunk)
                    if count > expected or time.monotonic() - start > 300:
                        raise ValueError("Download exceeds declared size or time budget")
                    output.write(chunk)
        if count != expected:
            raise ValueError("Downloaded size differs from Hub metadata")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_assets(kinds=tuple(SPECS), *, cache_dir=None, allow_downloads=None, max_bytes=MAX_BYTES) -> dict[str, Path]:
    """Prepare all requested snapshots after checking their aggregate size.

    Existing prepared assets count against the budget. Partial downloads are not
    considered valid caches. One cache-wide lock prevents competing preparations.
    """
    kinds = tuple(dict.fromkeys(kinds))
    if not kinds or any(kind not in SPECS for kind in kinds):
        raise ValueError("Choose bert, encoder and/or generator")
    if not isinstance(max_bytes, int) or not 0 < max_bytes <= MAX_BYTES:
        raise ValueError("Budget must be positive and at most 600,000,000 bytes")
    root = cache_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "models.prepare.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Asset preparation already locked: {lock}; remove only after checking no preparation is running") from exc
    os.close(fd)
    try:
        found = {kind: _cached(kind, root) for kind in SPECS}
        result = {kind: found[kind] for kind in kinds if found[kind] is not None}
        missing = [kind for kind in kinds if kind not in result]
        allowed = os.environ.get("AI_LAB_ALLOW_DOWNLOADS") == "1" if allow_downloads is None else allow_downloads is True
        if missing and not allowed:
            raise FileNotFoundError("Model assets missing; downloads require explicit --allow-downloads")
        plans = {kind: _plan(kind) for kind in missing}
        # Count all bytes already present, including interrupted downloads, to
        # enforce a physical cache ceiling as well as the selected payload ceiling.
        used = sum(p.stat().st_size for p in (root / "models").rglob("*") if p.is_file())
        # Reserve manifest/pointer bytes before any model download. The actual
        # metadata is smaller than this allowance for the fixed filename set.
        reusable = {}
        for kind, plan in plans.items():
            snapshot = root / "models" / kind / plan["revision"]
            reusable[kind] = {name for name, size in plan["sizes"].items()
                              if _matches_upstream(snapshot / name, size, plan["upstream_hashes"][name])}
        required = sum(16_384 + sum(size for name, size in plan["sizes"].items()
                                   if name not in reusable[kind]) for kind, plan in plans.items())
        if used + required > max_bytes:
            raise ValueError(f"Aggregate model cache budget exceeded: {used + required} > {max_bytes}")
        for kind, plan in plans.items():
            path = root / "models" / kind / plan["revision"]
            path.mkdir(parents=True, exist_ok=True)
            details = {}
            for name, size in plan.pop("sizes").items():
                target = path / name
                url = f"https://huggingface.co/{plan['repo_id']}/resolve/{plan['revision']}/{quote(name)}"
                if name not in reusable[kind]:
                    _download(url, target, size)
                if not _matches_upstream(target, size, plan["upstream_hashes"][name]):
                    raise ValueError("Actual asset size or upstream hash mismatch")
                details[name] = {"size": size, "sha256": _hash(target)}
            plan["files"] = details
            plan["actual_bytes"] = sum(item["size"] for item in details.values())
            _write_json(path / "metadata.json", plan)
            asset_metadata(path)
            _write_json(path.parent / "current.json", {"revision": plan["revision"]})
            result[kind] = path
        actual = sum(p.stat().st_size for p in (root / "models").rglob("*") if p.is_file())
        if actual > max_bytes:
            raise ValueError("Actual model cache exceeds aggregate byte budget")
        return result
    finally:
        lock.unlink(missing_ok=True)
