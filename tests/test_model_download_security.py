import hashlib
import os
from pathlib import Path
from typing import Optional
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yolo26_app.core.auto_annotator import SAM2_MODEL_SHA256, SAMAnnotator
from yolo26_app.ui.annotation import _ModelDownloadWorker


class _FakeResponse:
    def __init__(
        self,
        data: bytes,
        url: str,
        content_length: Optional[int] = None,
    ) -> None:
        self._data = data
        self._url = url
        self._offset = 0
        length = len(data) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def geturl(self) -> str:
        return self._url

    def read(self, size: int) -> bytes:
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_model_download_verifies_hash_before_atomic_replace(tmp_path: Path):
    data = b"trusted-sam-model"
    digest = hashlib.sha256(data).hexdigest()
    url = "https://dl.fbaipublicfiles.com/model.pt"
    destination = tmp_path / "model.pt"
    completed = []
    errors = []
    worker = _ModelDownloadWorker(url, str(destination), digest)
    worker.completed.connect(completed.append)
    worker.error.connect(errors.append)

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(data, url),
    ):
        worker.run()

    assert destination.read_bytes() == data
    assert completed == [str(destination)]
    assert errors == []
    assert not Path(f"{destination}.tmp").exists()


def test_model_download_removes_partial_file_on_hash_mismatch(tmp_path: Path):
    data = b"tampered"
    url = "https://dl.fbaipublicfiles.com/model.pt"
    destination = tmp_path / "model.pt"
    errors = []
    worker = _ModelDownloadWorker(url, str(destination), "0" * 64)
    worker.error.connect(errors.append)

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(data, url),
    ):
        worker.run()

    assert not destination.exists()
    assert not Path(f"{destination}.tmp").exists()
    assert errors and "完整性校验失败" in errors[0]


def test_model_download_rejects_untrusted_host_without_network_access(tmp_path: Path):
    destination = tmp_path / "model.pt"
    worker = _ModelDownloadWorker(
        "https://example.com/model.pt",
        str(destination),
        "0" * 64,
    )
    errors = []
    worker.error.connect(errors.append)

    with patch("urllib.request.urlopen") as urlopen:
        worker.run()

    urlopen.assert_not_called()
    assert errors and "受信任" in errors[0]
    assert not destination.exists()


def test_model_download_rejects_oversized_content_length(tmp_path: Path):
    data = b"x"
    url = "https://dl.fbaipublicfiles.com/model.pt"
    destination = tmp_path / "model.pt"
    worker = _ModelDownloadWorker(
        url,
        str(destination),
        hashlib.sha256(data).hexdigest(),
        max_bytes=4,
    )
    errors = []
    worker.error.connect(errors.append)

    with patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(data, url, content_length=5),
    ):
        worker.run()

    assert errors and "最大大小" in errors[0]
    assert not destination.exists()


def test_existing_official_model_file_is_hash_verified(tmp_path: Path, monkeypatch):
    data = b"existing-model"
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(data)
    monkeypatch.setitem(
        SAM2_MODEL_SHA256,
        "test-model",
        hashlib.sha256(data).hexdigest(),
    )

    assert SAMAnnotator.verify_official_model_file(str(model_path), "test-model")

    model_path.write_bytes(b"tampered")
    assert not SAMAnnotator.verify_official_model_file(str(model_path), "test-model")
