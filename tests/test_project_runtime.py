"""Runtime contracts use synthetic fixtures; these scores are not UCI evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from ai_lab.projects import (
    SMSDataset, SMSRecord, SMSSplits, TfidfFeatures, canonical_sha256,
    classification_metrics, downloads_allowed, evaluate_sms, load_sms_artifact,
    load_sms_records, load_sms_splits, make_sms_splits, predict_sms,
    save_sms_artifact, save_sms_splits, train_sms_model,
)
from ai_lab.projects.sms import _apportion


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    lines = [f"ham\tMeet me at lunch tomorrow example {i}" for i in range(60)]
    lines += [f"spam\tWin a free cash prize today example {i}" for i in range(20)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return load_sms_records(path)


@pytest.fixture
def splits(dataset):
    return make_sms_splits(dataset, train_size=40, validation_size=20, test_size=20)


@pytest.fixture
def artifact(splits):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield train_sms_model(splits, epochs=4, batch_size=8, hidden_dim=8, min_df=1)
    finally:
        torch.set_num_threads(previous)


def test_parsing_preserves_text_physical_ids_and_excludes_conflicts(tmp_path):
    source = tmp_path / "messages"
    source.write_text("ham\t Hello\tWORLD  \nspam\tＰＲＩＺＥ\nham\thello world\n"
                      "ham\tprize\nspam\tUnique offer\n", encoding="utf-8")
    data = load_sms_records(source)
    assert [(r.record_id, r.text, r.label) for r in data.records] == [
        ("sms-000001", " Hello\tWORLD  ", 0), ("sms-000005", "Unique offer", 1)]
    assert data.metadata["duplicate_rows"] == 2
    assert data.metadata["conflicting_canonical_sha256"] == [canonical_sha256("prize")]
    assert data.metadata["source"]["name"] == "Local SMS-format override"


@pytest.mark.parametrize("payload", [b"ham no tab", b"other\ttext", b"ham\t  ", b"ham\tbad\xff", b"\n", b""])
def test_malformed_sources_fail(tmp_path, payload):
    path = tmp_path / "messages"
    path.write_bytes(payload)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        load_sms_records(path)


def test_zip_member_read_never_extracts_paths(tmp_path):
    archive = tmp_path / "sms.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("../../outside.txt", "not extracted")
        stream.writestr("SMSSpamCollection", "ham\tSafe message\nspam\tOffer today\n")
    assert len(load_sms_records(archive).records) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sms.zip"]
    with ZipFile(archive, "w") as stream:
        stream.writestr("../SMSSpamCollection", "ham\twrong member")
    with pytest.raises(ValueError, match="one bounded"):
        load_sms_records(archive)


def test_download_opt_in_is_exact_and_explicit_override_does_not_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_LAB_SMS_PATH", raising=False)
    monkeypatch.setenv("AI_LAB_ALLOW_DOWNLOADS", "true")
    assert not downloads_allowed()
    with pytest.raises(FileNotFoundError, match="opt in"):
        load_sms_records(cache_dir=tmp_path)
    monkeypatch.setenv("AI_LAB_ALLOW_DOWNLOADS", "1")
    assert downloads_allowed()
    with pytest.raises(FileNotFoundError, match="override"):
        load_sms_records(tmp_path / "missing", allow_downloads=True)


def test_cache_is_used_without_network(tmp_path, monkeypatch):
    monkeypatch.delenv("AI_LAB_SMS_PATH", raising=False)
    cached = tmp_path / "sms" / "sms-spam-collection.zip"
    cached.parent.mkdir()
    raw = b"ham\tLocal fixture\nspam\tAnother fixture\n"
    with ZipFile(cached, "w") as stream:
        stream.writestr("SMSSpamCollection", raw)
    monkeypatch.setattr("ai_lab.projects.sms.urlopen", lambda *a, **k: pytest.fail("Unexpected network"))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_sms_records(cache_dir=tmp_path, allow_downloads=False)
    from hashlib import sha256
    monkeypatch.setattr("ai_lab.projects.sms.SMS_RAW_SHA256", sha256(raw).hexdigest())
    assert len(load_sms_records(cache_dir=tmp_path, allow_downloads=False).records) == 2


def test_split_exact_quota_seed_order_and_disjointness(dataset, splits):
    assert _apportion(3, {0: 2, 1: 2}) == {0: 2, 1: 1}
    assert splits.metadata["class_counts"] == {"train": {"0": 30, "1": 10},
                                               "validation": {"0": 15, "1": 5},
                                               "test": {"0": 15, "1": 5}}
    shuffled = SMSDataset(tuple(reversed(dataset.records)), dataset.metadata)
    assert make_sms_splits(shuffled, train_size=40, validation_size=20, test_size=20) == splits
    assert len({r.record_id for rows in (splits.train, splits.validation, splits.test) for r in rows}) == 80
    from hashlib import sha256
    ordered = sorted([r for r in dataset.records if r.label == 1],
                     key=lambda r: (sha256(f"42|{r.record_id}".encode()).digest(), r.record_id))
    assert {r.record_id for r in splits.test if r.label == 1} == {r.record_id for r in ordered[:5]}
    with pytest.raises(ValueError, match="Insufficient"):
        make_sms_splits(dataset)


def test_saved_splits_reuse_ids_and_detect_source_and_text_corruption(tmp_path, splits):
    path = save_sms_splits(splits, tmp_path / "prepared")
    assert load_sms_splits(path) == splits
    with pytest.raises(FileExistsError):
        save_sms_splits(splits, path)
    original = (path / "source.json").read_bytes()
    (path / "source.json").write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash"):
        load_sms_splits(path)
    (path / "source.json").write_bytes(original)
    with (path / "records.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="hash"):
        load_sms_splits(path)


def test_duplicate_text_cannot_leak_despite_different_ids(dataset):
    row = dataset.records[0]
    bad = SMSDataset(dataset.records + (SMSRecord("sms-999999", row.text.upper(), row.label),), dataset.metadata)
    with pytest.raises(ValueError, match="canonical"):
        make_sms_splits(bad)


def test_tfidf_training_only_math_oov_and_json_roundtrip():
    features = TfidfFeatures.fit(["apple apple pear", "pear orange"], min_df=1)
    assert features.vocabulary == ("pear", "apple", "orange")
    assert features.idf[0] == 1.0
    assert features.idf[1] == pytest.approx(np.log(3 / 2) + 1)
    assert "heldout" not in features.vocabulary
    transformed = features.transform(["apple pear", "heldout"])
    assert transformed.dtype == np.float32
    assert np.linalg.norm(transformed[0]) == pytest.approx(1)
    assert np.count_nonzero(transformed[1]) == 0
    restored = TfidfFeatures.from_dict(json.loads(json.dumps(features.to_dict())))
    np.testing.assert_array_equal(restored.transform(["apple pear", "heldout"]), transformed)


def test_metrics_confusion_and_zero_recall():
    result = classification_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert result["confusion"] == [[1, 1], [1, 1]]
    assert result["macro_f1"] == 0.5
    majority = classification_metrics([0, 0, 1], [0, 0, 0])
    assert majority["spam_precision"] == majority["spam_recall"] == 0
    assert majority["accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        classification_metrics([0], [])


def test_training_is_real_minibatched_and_does_not_fit_heldout(artifact, splits):
    history = artifact.metadata["training"]["history"]
    assert len(history) == 4
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert artifact.features.n_documents == len(splits.train)
    assert "test" not in artifact.metadata
    assert artifact.metadata["baseline"]["validation"]["confusion"] == [[15, 0], [5, 0]]
    assert artifact.metadata["validation"]["metrics"]["n"] == 20
    assert evaluate_sms(artifact, splits.test)["metrics"]["n"] == 20


def test_training_rejects_leakage_even_with_new_ids(splits):
    duplicate = SMSRecord("sms-999999", splits.train[0].text.upper(), splits.train[0].label)
    leaked = SMSSplits(splits.train, (duplicate,) + splits.validation[1:], splits.test, splits.metadata)
    with pytest.raises(ValueError, match="overlap"):
        train_sms_model(leaked, epochs=1)


def test_prediction_routes_are_separate_and_saved_policy_is_reused(artifact, tmp_path):
    for parameter in artifact.model.parameters():
        with torch.no_grad():
            parameter.zero_()
    artifact.metadata["routing_policy"]["review_threshold"] = 0.8
    artifact.metadata["routing_policy"]["selection"] = "synthetic test policy"
    path = save_sms_artifact(artifact, tmp_path / "router-v1")
    restored = load_sms_artifact(path)
    result = predict_sms(restored, ["Unseen words"])[0]
    assert result["label"] == "ham" and result["route"] == "review"
    assert result["probabilities"] == {"ham": 0.5, "spam": 0.5}
    assert predict_sms(restored, ["Unseen words"], review_threshold=0)[0]["route"] == "deliver"
    assert predict_sms(restored, []) == []
    with pytest.raises(ValueError):
        predict_sms(restored, [" "])
    with pytest.raises(TypeError):
        predict_sms(restored, "not a batch")


def test_artifact_corruption_version_and_path_rejection(artifact, tmp_path):
    path = save_sms_artifact(artifact, tmp_path / "model")
    with pytest.raises(FileExistsError):
        save_sms_artifact(artifact, path)
    manifest_path = path / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    data = json.loads(original)
    data["schema_version"] = 99
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_sms_artifact(path)
    data = json.loads(original)
    data["files"]["../outside"] = "arbitrary"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="paths"):
        load_sms_artifact(path)
    manifest_path.write_text(original, encoding="utf-8")
    with (path / "weights.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_sms_artifact(path)


def test_fresh_process_restores_without_data_network_or_training(artifact, tmp_path):
    path = save_sms_artifact(artifact, tmp_path / "restored")
    texts = ["Meet for lunch tomorrow", "Free prize today"]
    expected = predict_sms(artifact, texts)
    script = """
import json, socket, sys, torch
import ai_lab.projects.sms as runtime
def prohibited(*args, **kwargs):
    raise AssertionError('restoration attempted network, data ingestion, or training')
socket.socket = prohibited
runtime.load_sms_records = prohibited
runtime.train_sms_model = prohibited
print(json.dumps(runtime.predict_sms(runtime.load_sms_artifact(sys.argv[1]), json.loads(sys.argv[2]))))
"""
    env = os.environ.copy()
    env.pop("SSLKEYLOGFILE", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["AI_LAB_ALLOW_DOWNLOADS"] = "0"
    env["AI_LAB_SMS_PATH"] = str(tmp_path / "does-not-exist")
    result = subprocess.run([sys.executable, "-c", script, str(path), json.dumps(texts)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert [r["label"] for r in actual] == [r["label"] for r in expected]
    for left, right in zip(actual, expected):
        assert left["artifact_id"] == right["artifact_id"]
        assert left["confidence"] == pytest.approx(right["confidence"], abs=1e-6)
