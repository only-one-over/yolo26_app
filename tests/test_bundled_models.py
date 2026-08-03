from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from yolo26_app.core import bundled_models as bundled
from yolo26_app.core.auto_annotator import SAMAnnotator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = PROJECT_ROOT / "tools" / "fetch_pretrained_assets.py"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_pretrained_assets", FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_manifest(root: Path, digest: str) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "name": "yolo26n.pt",
                        "destination": "yolo/yolo26n.pt",
                        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_manifest_rejects_untrusted_urls_and_invalid_hashes(tmp_path: Path):
    module = _load_fetch_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "name": "bad.pt",
                        "destination": "../bad.pt",
                        "url": "https://example.com/bad.pt",
                        "sha256": "not-a-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        module.load_manifest(manifest)


def test_bundled_model_installs_once_and_preserves_existing_user_file(tmp_path: Path, monkeypatch):
    assets = tmp_path / "assets"
    source = assets / "yolo" / "yolo26n.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"bundled-model")
    _write_manifest(assets, hashlib.sha256(source.read_bytes()).hexdigest())
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(bundled, "_asset_root", lambda: assets)
    monkeypatch.setattr(bundled, "SYSTEM_MODEL_SUBDIRS", {"yolo": workspace / "yolo", "sam2": workspace / "sam2"})

    model = bundled.bundled_models()[0]
    assert bundled.resolve_model_path(workspace / "yolo", "yolo26n.pt") == source
    assert bundled.install_bundled_models() == (1, 0)
    destination = workspace / "yolo" / "yolo26n.pt"
    assert destination.read_bytes() == b"bundled-model"

    destination.write_bytes(b"user-model")
    assert bundled.install_bundled_models([model]) == (0, 1)
    assert destination.read_bytes() == b"user-model"


def test_bundled_model_checksum_failure_does_not_create_destination(tmp_path: Path, monkeypatch):
    assets = tmp_path / "assets"
    source = assets / "yolo" / "yolo26n.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"tampered")
    _write_manifest(assets, "0" * 64)
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(bundled, "_asset_root", lambda: assets)
    monkeypatch.setattr(bundled, "SYSTEM_MODEL_SUBDIRS", {"yolo": workspace / "yolo", "sam2": workspace / "sam2"})

    with pytest.raises(ValueError, match="checksum mismatch"):
        bundled.install_bundled_models()
    assert not (workspace / "yolo" / "yolo26n.pt").exists()


def test_sam2_config_is_resolved_from_bundled_resource_package(tmp_path: Path, monkeypatch):
    package_dir = tmp_path / "sam2_configs" / "sam2.1"
    package_dir.mkdir(parents=True)
    (package_dir.parent / "__init__.py").write_text("", encoding="utf-8")
    config = package_dir / "sam2.1_hiera_t.yaml"
    config.write_text("model: tiny\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("sam2_configs", None)
    importlib.invalidate_caches()

    annotator = SAMAnnotator()
    annotator._sam2_package_dir = None

    assert Path(annotator.resolve_config_path("configs/sam2.1/sam2.1_hiera_t.yaml")) == config
