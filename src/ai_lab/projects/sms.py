"""Small, inspectable SMS classification runtime shared by the book projects.

Data preparation uses only the standard library. NumPy/PyTorch are imported at
the feature/model boundary so reading the corpus needs no ML installation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.request import urlopen
from zipfile import ZipFile, BadZipFile

TASK_VERSION = "sms-routing.v1"
LABEL_MAP = {"ham": 0, "spam": 1}
SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# Observed from the official archive on 2026-10-28; fail on silently changed cache/source.
SMS_RAW_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
SOURCE = {
    "name": "UCI SMS Spam Collection",
    "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
    "download_url": SMS_URL,
    "doi": "10.24432/C5CC84",
    "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository.",
    "license": "CC-BY-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
}
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_RECORDS = 20_000
MAX_TEXT_CHARS = 10_000


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def canonical_text(text: str) -> str:
    """Identity normalization, also used by features: NFKC, casefold, whitespace."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text exceeds {MAX_TEXT_CHARS} characters")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    if not normalized:
        raise ValueError("text must not be blank")
    return normalized


def canonical_sha256(text: str) -> str:
    return sha256(canonical_text(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SMSRecord:
    record_id: str
    text: str
    label: int


@dataclass(frozen=True)
class SMSDataset:
    records: tuple[SMSRecord, ...]
    metadata: dict


@dataclass(frozen=True)
class SMSSplits:
    train: tuple[SMSRecord, ...]
    validation: tuple[SMSRecord, ...]
    test: tuple[SMSRecord, ...]
    metadata: dict


def _parse_records(raw: bytes, source: dict) -> SMSDataset:
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("SMS file exceeds the bounded source size")
    text = raw.decode("utf-8-sig", errors="strict")
    lines = text.splitlines()
    if not lines or len(lines) > MAX_RECORDS:
        raise ValueError("SMS file must contain 1..20000 records")
    records = {}
    duplicate_count = 0
    conflicts = set()
    for line_number, line in enumerate(lines, 1):
        try:
            label_name, message = line.split("\t", 1)
            label = LABEL_MAP[label_name]
            identity = canonical_sha256(message)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid SMS record at line {line_number}") from exc
        candidate = SMSRecord(f"sms-{line_number:06d}", message, label)
        if identity in records:
            duplicate_count += 1
            if records[identity].label != label:
                conflicts.add(identity)
        else:
            records[identity] = candidate
    # Conflicting labels cannot be resolved without new supervision. Quarantine
    # the entire identity, rather than choosing whichever row appeared first.
    for identity in conflicts:
        del records[identity]
    result = tuple(sorted(records.values(), key=lambda r: r.record_id))
    if not result:
        raise ValueError("No records remain after conflicting-label quarantine")
    metadata = {
        "task_version": TASK_VERSION, "source": dict(source),
        "raw_sha256": sha256(raw).hexdigest(), "raw_rows": len(lines),
        "unique_rows": len(result), "duplicate_rows": duplicate_count,
        "conflicting_canonical_sha256": sorted(conflicts),
        "identity_policy": "physical-line-1based.v1",
        "canonicalization": "nfkc-casefold-whitespace.v1",
        "label_map": dict(LABEL_MAP),
    }
    metadata["dataset_sha256"] = sha256(_json_bytes([
        [r.record_id, r.label, canonical_sha256(r.text)] for r in result])).hexdigest()
    return SMSDataset(result, metadata)


def _read_source(path: Path) -> bytes:
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Source file exceeds 4 MiB")
    payload = path.read_bytes()
    if payload.startswith(b"PK"):
        try:
            with ZipFile(BytesIO(payload)) as archive:
                members = [x for x in archive.infolist() if x.filename == "SMSSpamCollection"]
                if len(members) != 1 or members[0].file_size > MAX_SOURCE_BYTES:
                    raise ValueError("Archive must contain one bounded SMSSpamCollection member")
                # No extract/extractall: archive names never become filesystem paths.
                with archive.open(members[0]) as stream:
                    raw = stream.read(MAX_SOURCE_BYTES + 1)
                if len(raw) > MAX_SOURCE_BYTES:
                    raise ValueError("Expanded SMS member exceeds 4 MiB")
                return raw
        except BadZipFile as exc:
            raise ValueError("Invalid SMS ZIP archive") from exc
    return payload


def cache_directory(cache_dir=None) -> Path:
    return Path(cache_dir or os.environ.get("AI_LAB_CACHE_DIR", ".cache/ai_lab"))


def downloads_allowed(allow_downloads=None) -> bool:
    if allow_downloads is not None:
        if not isinstance(allow_downloads, bool):
            raise TypeError("allow_downloads must be a bool or None")
        return allow_downloads
    return os.environ.get("AI_LAB_ALLOW_DOWNLOADS", "") == "1"


def load_sms_records(path=None, *, allow_downloads=None, cache_dir=None) -> SMSDataset:
    """Read a local file/ZIP, then cache; download only with explicit opt-in.

    AI_LAB_SMS_PATH has precedence over cache and must exist when supplied.
    Local overrides record their own origin; the runtime cannot authenticate
    arbitrary local bytes as UCI data. Unit fixtures never become UCI evidence.
    """
    override = path or os.environ.get("AI_LAB_SMS_PATH")
    cached = cache_directory(cache_dir) / "sms" / "sms-spam-collection.zip"
    selected = Path(override) if override else cached
    if override and not selected.is_file():
        raise FileNotFoundError(f"SMS override does not exist: {selected}")
    if not selected.is_file():
        if not downloads_allowed(allow_downloads):
            raise FileNotFoundError("SMS data missing. Set AI_LAB_SMS_PATH to a local SMSSpamCollection/ZIP, "
                                    "or opt in with AI_LAB_ALLOW_DOWNLOADS=1.")
        selected.parent.mkdir(parents=True, exist_ok=True)
        # Bounded transfer, no retry loop, no archive extraction.
        with urlopen(SMS_URL, timeout=30) as response:
            payload = response.read(MAX_SOURCE_BYTES + 1)
        if len(payload) > MAX_SOURCE_BYTES:
            raise ValueError("SMS download exceeds 4 MiB")
        fd, tmp_name = tempfile.mkstemp(prefix="sms-", suffix=".tmp", dir=selected.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            downloaded = _parse_records(_read_source(Path(tmp_name)), SOURCE)
            if downloaded.metadata["raw_sha256"] != SMS_RAW_SHA256:
                raise ValueError("UCI source identity changed; verify before updating the pinned source hash")
            os.replace(tmp_name, selected)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
    provenance = dict(SOURCE) if not override else {
        "name": "Local SMS-format override", "origin": str(selected.resolve()),
        "license": "caller-supplied; verify before redistribution",
        "expected_public_source": dict(SOURCE),
    }
    result = _parse_records(_read_source(selected), provenance)
    if not override and result.metadata["raw_sha256"] != SMS_RAW_SHA256:
        raise ValueError("Cached UCI source identity mismatch; verify or replace the cache")
    result.metadata["verified_public_source"] = not bool(override)
    result.metadata["source_file_sha256"] = sha256(selected.read_bytes()).hexdigest()
    result.metadata["loaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    result.metadata["modifications"] = "Original text retained; canonical duplicates removed; conflicting groups excluded."
    return result


def make_sms_splits(dataset: SMSDataset, *, train_size=800, validation_size=200,
                    test_size=200, seed=42) -> SMSSplits:
    """Proportional stratification after deduplication with persisted physical IDs.

    Each class is ordered by SHA256(seed|record_id). Largest-remainder allocation
    selects the subset, then test, then validation; train uses the remainder.
    Unused rows remain outside the teaching subset. This is a random split, not
    temporal or sender-disjoint; it does not establish deployment generalization.
    """
    sizes = (train_size, validation_size, test_size)
    if any(type(x) is not int or x < 2 for x in sizes) or type(seed) is not int:
        raise ValueError("Split sizes must be integers >=2 and seed an integer")
    if (len({r.record_id for r in dataset.records}) != len(dataset.records) or
            len({canonical_sha256(r.text) for r in dataset.records}) != len(dataset.records)):
        raise ValueError("Deduplicate record IDs and canonical texts before splitting")
    buckets = {label: [] for label in LABEL_MAP.values()}
    for row in dataset.records:
        if row.label not in buckets or not re.fullmatch(r"sms-\d{6}", row.record_id):
            raise ValueError("Invalid record identity or label")
        buckets[row.label].append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: (sha256(f"{seed}|{row.record_id}".encode()).digest(), row.record_id))
    available = {label: len(rows) for label, rows in buckets.items()}
    if sum(sizes) > sum(available.values()):
        raise ValueError("Insufficient records for requested split sizes")
    remaining = _apportion(sum(sizes), available)
    partitions = {}
    offsets = {0: 0, 1: 0}
    for name, size in (("test", test_size), ("validation", validation_size), ("train", train_size)):
        quotas = _apportion(size, remaining)
        if any(count < 1 for count in quotas.values()):
            raise ValueError("Insufficient class coverage: each split needs both classes")
        selected = []
        for label, count in quotas.items():
            selected.extend(buckets[label][offsets[label]:offsets[label] + count])
            offsets[label] += count
            remaining[label] -= count
        partitions[name] = tuple(sorted(selected, key=lambda row: row.record_id))
    ids = {name: [row.record_id for row in partitions[name]] for name in ("train", "validation", "test")}
    metadata = {"task_version": TASK_VERSION, "source_metadata": dataset.metadata,
                "split_policy": "sha256-largest-remainder-test-first.v1", "seed": seed,
                "sizes": dict(zip(ids, sizes)), "ids": ids,
                "class_counts": {name: {str(label): sum(r.label == label for r in rows)
                                        for label in (0, 1)}
                                 for name, rows in partitions.items()}}
    metadata["split_sha256"] = sha256(_json_bytes(ids)).hexdigest()
    return SMSSplits(partitions["train"], partitions["validation"], partitions["test"], metadata)


def _apportion(size, counts):
    total = sum(counts.values())
    if total < size or total == 0:
        raise ValueError("Insufficient records for stratified allocation")
    quotas = {label: size * count // total for label, count in counts.items()}
    order = sorted(counts, key=lambda label: (-(size * counts[label] % total), label))
    for label in order[:size - sum(quotas.values())]:
        quotas[label] += 1
    return quotas


def save_sms_splits(splits: SMSSplits, path):
    """Persist selected original texts and frozen IDs so later chapters do not split again."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Choose a new prepared-data directory: {destination}")
    records = []
    for name in ("train", "validation", "test"):
        rows = getattr(splits, name)
        if [r.record_id for r in rows] != splits.metadata["ids"][name]:
            raise ValueError("Rows differ from persisted split IDs")
        records.extend({"id": r.record_id, "text": r.text, "label": ("ham", "spam")[r.label],
                        "label_id": r.label, "source_line": int(r.record_id[4:]),
                        "text_sha256": canonical_sha256(r.text), "split": name,
                        "task_id": TASK_VERSION} for r in rows)
    if len({r["text_sha256"] for r in records}) != len(records):
        raise ValueError("Canonical texts overlap across prepared data")
    payload = b"".join(_json_bytes(row) + b"\n" for row in records)
    source = _json_bytes(splits.metadata["source_metadata"])
    manifest = dict(splits.metadata)
    manifest.update({"schema_version": 1, "records_sha256": sha256(payload).hexdigest(),
                     "source_sha256": sha256(source).hexdigest()})
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + "-", dir=destination.parent))
    try:
        for name, content in (("records.jsonl", payload), ("source.json", source),
                              ("splits.json", _json_bytes(manifest))):
            (staging / name).write_bytes(content)
        staging.rename(destination)
    finally:
        if staging.exists():
            for name in ("records.jsonl", "source.json", "splits.json"):
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    return destination


def load_sms_splits(path) -> SMSSplits:
    """Read persisted IDs/order, verify source/record hashes, and reject split leakage."""
    directory = Path(path)
    payloads = {}
    for name in ("records.jsonl", "source.json", "splits.json"):
        file = directory / name
        if file.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"Prepared file exceeds 16 MiB: {name}")
        payloads[name] = file.read_bytes()
    metadata = json.loads(payloads["splits.json"])
    source = json.loads(payloads["source.json"])
    if (metadata.get("schema_version") != 1 or metadata.get("task_version") != TASK_VERSION or
            metadata.get("source_sha256") != sha256(payloads["source.json"]).hexdigest() or
            metadata.get("records_sha256") != sha256(payloads["records.jsonl"]).hexdigest() or
            metadata.get("source_metadata") != source or
            source.get("label_map") != LABEL_MAP):
        raise ValueError("Prepared data schema/source/hash mismatch")
    ids = metadata["ids"]
    if (set(ids) != {"train", "validation", "test"} or
            metadata["split_sha256"] != sha256(_json_bytes(ids)).hexdigest()):
        raise ValueError("Prepared split identity mismatch")
    records, seen_texts = {}, set()
    for line in payloads["records.jsonl"].splitlines():
        row = json.loads(line)
        identity = row["id"]
        digest = canonical_sha256(row["text"])
        if (identity in records or digest in seen_texts or row["task_id"] != TASK_VERSION or
                not re.fullmatch(r"sms-\d{6}", identity) or row["source_line"] != int(identity[4:]) or
                row["text_sha256"] != digest or LABEL_MAP.get(row["label"]) != row["label_id"] or
                row["split"] not in ids or identity not in ids[row["split"]]):
            raise ValueError("Invalid or duplicate prepared record")
        records[identity] = SMSRecord(identity, row["text"], row["label_id"])
        seen_texts.add(digest)
    flattened = [identity for name in ids for identity in ids[name]]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(records):
        raise ValueError("Prepared split IDs overlap or differ from records")
    partitions = {name: tuple(records[identity] for identity in ids[name]) for name in ids}
    for name, rows in partitions.items():
        counts = {str(label): sum(r.label == label for r in rows) for label in (0, 1)}
        if len(rows) != metadata["sizes"][name] or counts != metadata["class_counts"][name] or min(counts.values()) < 1:
            raise ValueError("Prepared split counts mismatch")
    # Preserve the in-memory schema so a saved/reloaded split compares directly.
    metadata = {k: v for k, v in metadata.items() if k not in {"schema_version", "records_sha256", "source_sha256"}}
    return SMSSplits(partitions["train"], partitions["validation"], partitions["test"], metadata)


@dataclass(frozen=True)
class TfidfFeatures:
    """Unigram TF-IDF with log term frequency and row L2 normalization.

    idf(t)=log((1+n)/(1+df(t)))+1; tf(t)=1+log(count(t)). Vocabulary
    ranking is descending document frequency then lexical order. No held-out
    text is used to fit either vocabulary or IDF. Out-of-vocabulary rows are zero.
    """
    vocabulary: tuple[str, ...]
    idf: tuple[float, ...]
    n_documents: int
    max_features: int = 2048
    min_df: int = 2

    @classmethod
    def fit(cls, texts, *, max_features=2048, min_df=2):
        if type(max_features) is not int or not 1 <= max_features <= 20_000:
            raise ValueError("max_features must be 1..20000")
        if type(min_df) is not int or min_df < 1:
            raise ValueError("min_df must be a positive integer")
        documents = list(texts)
        counts = Counter(token for text in documents for token in set(_tokens(text)))
        vocabulary = tuple(sorted((t for t, n in counts.items() if n >= min_df),
                                  key=lambda t: (-counts[t], t))[:max_features])
        if not vocabulary:
            raise ValueError("Empty vocabulary: lower min_df or inspect training data")
        n = len(documents)
        return cls(vocabulary, tuple(math.log((1+n)/(1+counts[t]))+1 for t in vocabulary),
                   n, max_features, min_df)

    def transform(self, texts):
        import numpy as np
        documents = list(texts)
        if len(documents) > MAX_RECORDS:
            raise ValueError("Feature batch exceeds 20000 records")
        matrix = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float32)
        lookup = {token: i for i, token in enumerate(self.vocabulary)}
        for row, text in enumerate(documents):
            for token, count in Counter(_tokens(text)).items():
                col = lookup.get(token)
                if col is not None:
                    matrix[row, col] = (1 + math.log(count)) * self.idf[col]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, np.finfo(np.float32).tiny)

    def to_dict(self):
        return {"format": "sms-tfidf.v1", "tokenizer": "unicode-word-2plus.v1",
                "canonicalization": "nfkc-casefold-whitespace.v1",
                "vocabulary": list(self.vocabulary), "idf": list(self.idf),
                "n_documents": self.n_documents, "max_features": self.max_features,
                "min_df": self.min_df}

    @classmethod
    def from_dict(cls, data):
        if (data.get("format") != "sms-tfidf.v1" or
                data.get("tokenizer") != "unicode-word-2plus.v1" or
                data.get("canonicalization") != "nfkc-casefold-whitespace.v1"):
            raise ValueError("Unsupported preprocessing format")
        vocabulary, idf = data["vocabulary"], data["idf"]
        if (not 1 <= len(vocabulary) <= 20_000 or len(set(vocabulary)) != len(vocabulary)
                or len(vocabulary) != len(idf)
                or any(not isinstance(t, str) or not t for t in vocabulary)
                or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 1 for v in idf)
                or type(data["n_documents"]) is not int or data["n_documents"] < 1):
            raise ValueError("Invalid preprocessing vocabulary/IDF")
        return cls(tuple(vocabulary), tuple(idf), data["n_documents"],
                   data["max_features"], data["min_df"])


def _tokens(text):
    return re.findall(r"\b\w\w+\b", canonical_text(text))


def SMSModel(config):
    """Build the exact small nn.Sequential architecture from a JSON config."""
    import torch.nn as nn
    if (config.get("architecture") != "tfidf-mlp.v1" or
            type(config.get("input_dim")) is not int or not 1 <= config["input_dim"] <= 20_000 or
            type(config.get("hidden_dim")) is not int or not 1 <= config["hidden_dim"] <= 512 or
            config.get("output_dim") != 2):
        raise ValueError("Invalid or unsupported model config")
    return nn.Sequential(nn.Linear(config["input_dim"], config["hidden_dim"]),
                         nn.ReLU(), nn.Linear(config["hidden_dim"], 2))


@dataclass
class SMSArtifact:
    model: object
    features: TfidfFeatures
    metadata: dict


def classification_metrics(labels, predictions):
    """Confusion rows=true, columns=predicted; zero-denominator scores are zero."""
    truth, predicted = list(labels), list(predictions)
    if not truth or len(truth) != len(predicted):
        raise ValueError("Expected nonempty equal-length labels and predictions")
    if any(type(x) is not int or x not in (0, 1) for x in truth + predicted):
        raise ValueError("Expected integer binary labels")
    confusion = [[sum(a == i and b == j for a, b in zip(truth, predicted))
                  for j in (0, 1)] for i in (0, 1)]
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    spam_f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    ham_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"n": len(truth), "confusion": confusion,
            "accuracy": (tp + tn) / len(truth), "spam_precision": precision,
            "spam_recall": recall, "spam_f1": spam_f1, "macro_f1": (spam_f1 + ham_f1) / 2,
            "balanced_accuracy": (recall + specificity) / 2}


def train_sms_model(splits: SMSSplits, *, seed=42, epochs=8, batch_size=32,
                    hidden_dim=32, max_features=2048, min_df=2, learning_rate=0.01):
    """Train on train only; report validation only. Test evaluation is explicit."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if type(epochs) is not int or not 1 <= epochs <= 100:
        raise ValueError("epochs must be 1..100")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("batch_size must be 1..4096")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be finite and in (0,1]")
    partitions = (splits.train, splits.validation, splits.test)
    all_ids = [r.record_id for part in partitions for r in part]
    text_ids = [canonical_sha256(r.text) for part in partitions for r in part]
    if len(all_ids) != len(set(all_ids)) or len(text_ids) != len(set(text_ids)):
        raise ValueError("Training/validation/test IDs or canonical texts overlap")
    if not splits.train or not splits.validation or set(r.label for r in splits.train) != {0, 1}:
        raise ValueError("Training needs both classes and nonempty validation")
    features = TfidfFeatures.fit([r.text for r in splits.train],
                                 max_features=max_features, min_df=min_df)
    x = torch.from_numpy(features.transform(r.text for r in splits.train))
    y = torch.tensor([r.label for r in splits.train], dtype=torch.long)
    config = {"architecture": "tfidf-mlp.v1", "input_dim": x.shape[1],
              "hidden_dim": hidden_dim, "output_dim": 2}
    history = []
    # Isolate initialization/dropout RNG state from the caller's notebook.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SMSModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            for inputs, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(inputs), labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(labels)
            history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(y)})
    model.eval()
    metadata = {"schema_version": 1, "artifact_format": "sms-classifier.v1",
                "task_version": TASK_VERSION, "task_id": TASK_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "label_map": dict(LABEL_MAP), "model_config": config,
                "split_metadata": splits.metadata,
                "training": {"seed": seed, "epochs": epochs, "batch_size": batch_size,
                             "learning_rate": learning_rate, "optimizer": "Adam",
                             "class_weighting": "none", "history": history},
                "versions": {"torch": str(torch.__version__)},
                "routing_policy": {"version": "routing-policy.v1", "review_threshold": 0.0,
                                   "selection": "mechanism default; not validation-selected"}}
    artifact = SMSArtifact(model, features, metadata)
    metadata["validation"] = evaluate_sms(artifact, splits.validation)
    majority = Counter(r.label for r in splits.train).most_common(1)[0][0]
    metadata["baseline"] = {"kind": "training-majority", "label_id": majority,
                            "validation": classification_metrics(
                                [r.label for r in splits.validation],
                                [majority] * len(splits.validation))}
    return artifact


def predict_sms(artifact: SMSArtifact, texts, *, review_threshold=None):
    """Return both the binary task prediction and a separate abstention route.

    Confidence is softmax probability, not calibrated certainty. Empty batches
    return []; blank/oversized messages are rejected. Batch limit is 4096.
    """
    import torch
    if review_threshold is None:
        review_threshold = artifact.metadata.get("routing_policy", {}).get("review_threshold", 0.0)
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")
    if not isinstance(review_threshold, (int, float)) or not math.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold must be finite in [0,1]")
    values = list(texts)
    if len(values) > 4096:
        raise ValueError("Prediction batch exceeds 4096 messages")
    if not values:
        return []
    artifact.model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(artifact.model(torch.from_numpy(artifact.features.transform(values))), dim=1)
    if probabilities.shape != (len(values), 2) or not torch.isfinite(probabilities).all():
        raise ValueError("Model returned invalid probabilities")
    result = []
    for row in probabilities.tolist():
        label_id = int(row[1] > row[0])
        confidence = row[label_id]
        label = ("ham", "spam")[label_id]
        result.append({"label": label, "label_id": label_id, "confidence": confidence,
                       "probabilities": dict(zip(("ham", "spam"), row)),
                       "route": "review" if confidence < review_threshold else ("deliver", "quarantine")[label_id],
                       "task_id": TASK_VERSION,
                       "policy_version": artifact.metadata.get("routing_policy", {}).get("version", "routing-policy.v1"),
                       "artifact_id": artifact.metadata.get("artifact_id", "unsaved")})
    return result


def evaluate_sms(artifact, records, *, error_limit=10):
    rows = list(records)
    if type(error_limit) is not int or not 0 <= error_limit <= 100:
        raise ValueError("error_limit must be 0..100")
    predictions = []
    for start in range(0, len(rows), 4096):
        predictions.extend(predict_sms(artifact, [r.text for r in rows[start:start+4096]]))
    metrics = classification_metrics([r.label for r in rows], [p["label_id"] for p in predictions])
    # IDs allow local inspection without copying public message contents into logs.
    errors = [{"record_id": r.record_id, "true_label": ("ham", "spam")[r.label],
               "predicted_label": p["label"], "confidence": p["confidence"]}
              for r, p in zip(rows, predictions) if r.label != p["label_id"]]
    return {"metrics": metrics, "errors": errors[:error_limit], "error_count": len(errors)}


def save_sms_artifact(artifact: SMSArtifact, path):
    """Write a new artifact directory; never overwrite an existing model run.

    JSON preprocessing and weights-only tensor state replace pickled estimators.
    Hashes detect accidental corruption, not an adversarial source's authenticity.
    """
    import torch
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Choose a new version directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + "-", dir=destination.parent))
    try:
        preprocessing = _json_bytes(artifact.features.to_dict())
        (staging / "preprocessing.json").write_bytes(preprocessing)
        torch.save({key: value.detach().cpu().clone() for key, value in artifact.model.state_dict().items()},
                   staging / "weights.pt")
        manifest = dict(artifact.metadata)
        manifest["files"] = {"preprocessing.json": sha256(preprocessing).hexdigest(),
                             "weights.pt": sha256((staging / "weights.pt").read_bytes()).hexdigest()}
        manifest["artifact_id"] = "sms-" + sha256(_json_bytes({k: v for k, v in manifest.items()
                                                              if k != "artifact_id"})).hexdigest()
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        staging.rename(destination)
        artifact.metadata.update(manifest)
    finally:
        # Delete only our explicitly enumerated temporary files, never a tree.
        if staging.exists():
            for name in ("preprocessing.json", "weights.pt", "manifest.json"):
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    return destination


def load_sms_artifact(path) -> SMSArtifact:
    """Restore CPU inference solely from a trusted artifact directory."""
    import torch
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Manifest exceeds 8 MiB")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("artifact_format") != "sms-classifier.v1" or
            manifest.get("task_version") != TASK_VERSION or manifest.get("label_map") != LABEL_MAP):
        raise ValueError("Incompatible artifact task, format, or label map")
    policy = manifest.get("routing_policy", {})
    threshold = policy.get("review_threshold")
    if (policy.get("version") != "routing-policy.v1" or not isinstance(threshold, (int, float)) or
            not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Invalid routing policy")
    if set(manifest.get("files", {})) != {"preprocessing.json", "weights.pt"}:
        raise ValueError("Unexpected artifact file paths")
    expected_id = "sms-" + sha256(_json_bytes({k: v for k, v in manifest.items() if k != "artifact_id"})).hexdigest()
    if manifest.get("artifact_id") != expected_id:
        raise ValueError("Manifest identity mismatch")
    payloads = {}
    for name, limit in (("preprocessing.json", 4 * 1024 * 1024), ("weights.pt", 64 * 1024 * 1024)):
        file = directory / name
        if file.stat().st_size > limit:
            raise ValueError(f"Artifact file too large: {name}")
        payloads[name] = file.read_bytes()
        if sha256(payloads[name]).hexdigest() != manifest.get("files", {}).get(name):
            raise ValueError(f"Artifact checksum mismatch: {name}")
    features = TfidfFeatures.from_dict(json.loads(payloads["preprocessing.json"]))
    config = manifest["model_config"]
    if config["input_dim"] != len(features.vocabulary):
        raise ValueError("Model/preprocessing dimension mismatch")
    model = SMSModel(config)
    state = torch.load(BytesIO(payloads["weights.pt"]), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Expected finite tensor-only model state")
    model.load_state_dict(state, strict=True)
    model.eval()
    return SMSArtifact(model, features, manifest)
