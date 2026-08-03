"""Resolve and install optional pretrained models bundled with CUDA-Full."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable

from PyQt6.QtCore import QThread, pyqtSignal

from yolo26_app.core.paths import SYSTEM_MODEL_SUBDIRS


@dataclass(frozen=True)
class BundledModel:
    name: str
    destination: str
    sha256: str


def _asset_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "pretrained"
    return Path(__file__).resolve().parents[3] / "assets" / "pretrained"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundled_models() -> list[BundledModel]:
    manifest_path = _asset_root() / "manifest.json"
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    result: list[BundledModel] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name, destination, digest = item.get("name"), item.get("destination"), item.get("sha256")
        source = _asset_root() / str(destination)
        if isinstance(name, str) and isinstance(destination, str) and isinstance(digest, str) and source.is_file():
            result.append(BundledModel(name, destination, digest))
    return result


def bundled_model_path(destination: str) -> Path | None:
    source = _asset_root() / destination
    return source if source.is_file() else None


def resolve_model_path(model_dir: Path, filename: str) -> Path:
    workspace_path = model_dir / filename
    if workspace_path.is_file():
        return workspace_path
    for model in bundled_models():
        if model.destination == f"{model_dir.name}/{filename}":
            source = bundled_model_path(model.destination)
            if source is not None:
                return source
    return workspace_path


def bundled_search_dirs(kind: str) -> list[str]:
    directories = [str(SYSTEM_MODEL_SUBDIRS[kind])]
    source = _asset_root() / kind
    if source.is_dir():
        directories.append(str(source))
    return directories


def install_bundled_models(
    models: Iterable[BundledModel] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    installed = 0
    skipped = 0
    for model in models if models is not None else bundled_models():
        if should_stop is not None and should_stop():
            break
        source = bundled_model_path(model.destination)
        if source is None:
            continue
        parts = Path(model.destination).parts
        if len(parts) != 2:
            continue
        kind, filename = parts
        if kind not in SYSTEM_MODEL_SUBDIRS:
            continue
        destination = SYSTEM_MODEL_SUBDIRS[kind] / filename
        if destination.exists():
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=destination.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
            if _sha256(temporary_path) != model.sha256:
                raise ValueError(f"Bundled model checksum mismatch: {model.name}")
            os.replace(temporary_path, destination)
            installed += 1
        finally:
            temporary_path.unlink(missing_ok=True)
    return installed, skipped


class BundledModelInstallWorker(QThread):
    completed = pyqtSignal(int, int)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            self.completed.emit(*install_bundled_models(should_stop=self.isInterruptionRequested))
        except Exception as exc:
            self.failed.emit(str(exc))
