"""Offline tests exercise acquisition policy; fixtures are not pretrained evidence."""
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_lab.projects import assets


@pytest.fixture(autouse=True)
def clean_asset_environment(monkeypatch):
    for name in ("AI_LAB_ALLOW_DOWNLOADS", "AI_LAB_CACHE_DIR", "AI_LAB_MODEL_DIR", "AI_LAB_ENCODER_DIR", "AI_LAB_GENERATOR_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def hub(monkeypatch):
    calls = []
    blobs = {"config.json": b'{"model_type":"bert"}', "vocab.txt": b"[UNK]\nhello\n",
             "README.md": b"---\nlicense: mit\n---\nfixture", "model.safetensors": b"fixtureweights"}
    def info(repo_id):
        calls.append(("metadata", repo_id))
        names = list(blobs) + ["pytorch_model.bin", "tf_model.h5", "onnx/model.onnx", "evil.py"]
        return SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename=n,
            size=len(blobs.get(n, b"extra")), blob_id=hashlib.sha1(
                f"blob {len(blobs.get(n, b'extra'))}\0".encode() + blobs.get(n, b"extra")).hexdigest()) for n in names])
    def download(url, path, expected):
        assert "/" + "a" * 40 + "/" in url
        calls.append(("download", path.name))
        path.write_bytes(blobs[path.name])
    monkeypatch.setattr(assets, "_model_info", info)
    monkeypatch.setattr(assets, "_download", download)
    return calls


def test_offline_default_and_exact_environment_opt_in(tmp_path, monkeypatch, hub):
    monkeypatch.setenv("AI_LAB_ALLOW_DOWNLOADS", "true")
    with pytest.raises(FileNotFoundError, match="Missing bert"):
        assets.resolve_asset("bert", cache_dir=tmp_path)
    assert hub == []
    monkeypatch.setenv("AI_LAB_ALLOW_DOWNLOADS", "1")
    assert assets.resolve_asset("bert", cache_dir=tmp_path).is_dir()


def test_one_weight_format_sha_provenance_and_offline_cache(tmp_path, hub, monkeypatch):
    path = assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)["bert"]
    meta = assets.asset_metadata(path)
    assert meta["revision"] == "a" * 40
    assert meta["repo_id"] == "prajjwal1/bert-tiny"
    assert meta["actual_bytes"] == meta["planned_bytes"]
    assert meta["trust_remote_code"] is False
    assert "model.safetensors" in meta["files"]
    assert "pytorch_model.bin" not in meta["files"]
    assert not (path / "evil.py").exists()
    monkeypatch.setattr(assets, "_model_info", lambda _: pytest.fail("Offline cache touched network"))
    assert assets.resolve_asset("bert", cache_dir=tmp_path) == path


def test_aggregate_budget_checked_before_any_download(tmp_path, hub):
    with pytest.raises(ValueError, match="budget"):
        assets.prepare_assets(cache_dir=tmp_path, allow_downloads=True, max_bytes=100)
    assert not any(call[0] == "download" for call in hub)
    assert not (tmp_path / "models.prepare.lock").exists()


def test_cached_models_count_toward_later_budget(tmp_path, hub):
    assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)
    hub.clear()
    with pytest.raises(ValueError, match="budget"):
        assets.prepare_assets(("encoder",), cache_dir=tmp_path, allow_downloads=True, max_bytes=100)
    assert not any(call[0] == "download" for call in hub)


def test_corruption_rejected_not_silently_redownloaded(tmp_path, hub):
    path = assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)["bert"]
    (path / "vocab.txt").write_bytes(b"corrupted")
    hub.clear()
    with pytest.raises(ValueError, match="integrity"):
        assets.resolve_asset("bert", cache_dir=tmp_path, allow_downloads=True)
    assert hub == []


def test_local_override_precedes_network(tmp_path, monkeypatch, hub):
    for name, data in {"config.json": "{}", "pytorch_model.bin": "fixture", "vocab.txt": "hello"}.items():
        (tmp_path / name).write_text(data)
    monkeypatch.setenv("AI_LAB_MODEL_DIR", str(tmp_path))
    assert assets.resolve_asset("bert", allow_downloads=True) == tmp_path
    assert hub == []
    (tmp_path / "vocab.txt").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        assets.resolve_asset("bert", allow_downloads=True)


@pytest.mark.parametrize("sha,size", [("main", 1), ("a" * 40, None), ("a" * 40, -1)])
def test_missing_immutable_revision_or_sizes_rejected(tmp_path, monkeypatch, sha, size):
    monkeypatch.setattr(assets, "_model_info", lambda _: SimpleNamespace(sha=sha, siblings=[
        SimpleNamespace(rfilename=name, size=size) for name in ("config.json", "README.md", "vocab.txt", "pytorch_model.bin")]))
    with pytest.raises(ValueError):
        assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)


def test_wrong_actual_bytes_are_not_published(tmp_path, monkeypatch, hub):
    monkeypatch.setattr(assets, "_download", lambda url, path, expected: path.write_bytes(b"wrong"))
    with pytest.raises(ValueError, match="size"):
        assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)
    assert not (tmp_path / "models" / "bert" / "current.json").exists()


def test_metadata_path_traversal_rejected(tmp_path, hub):
    path = assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)["bert"]
    meta = json.loads((path / "metadata.json").read_text())
    meta["files"]["../../outside"] = {"size": 0, "sha256": "a" * 64}
    (path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="filename"):
        assets.asset_metadata(path)


def test_invalid_budget_and_cache_lock(tmp_path):
    with pytest.raises(ValueError, match="Budget"):
        assets.prepare_assets(cache_dir=tmp_path, max_bytes=assets.MAX_BYTES + 1)
    (tmp_path / "models.prepare.lock").touch()
    with pytest.raises(RuntimeError, match="locked"):
        assets.prepare_assets(cache_dir=tmp_path)


def test_interrupted_download_reuses_only_upstream_verified_bytes(tmp_path, hub):
    path = assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)["bert"]
    (path.parent / "current.json").unlink()
    (path / "metadata.json").unlink()
    (path / "config.json").write_bytes(b"corrupted")
    hub.clear()
    assets.prepare_assets(("bert",), cache_dir=tmp_path, allow_downloads=True)
    assert [name for operation, name in hub if operation == "download"] == ["config.json"]
    assert assets.asset_metadata(path)["revision"] == "a" * 40
