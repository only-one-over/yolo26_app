"""Download the fixed CUDA-Full pretrained assets with SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "assets" / "pretrained" / "manifest.json"
ALLOWED_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "github-releases.githubusercontent.com",
    "dl.fbaipublicfiles.com",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Unsupported pretrained asset manifest.")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Pretrained asset manifest has no models.")
    result: list[dict[str, str]] = []
    destinations: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Invalid pretrained asset entry.")
        required = {key: item.get(key) for key in ("name", "destination", "url", "sha256")}
        if not all(isinstance(value, str) and value for value in required.values()):
            raise ValueError("Pretrained asset entry is missing required fields.")
        parsed = urlparse(required["url"])
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Untrusted pretrained asset URL: {required['url']}")
        destination = Path(required["destination"])
        if destination.is_absolute() or ".." in destination.parts or required["destination"] in destinations:
            raise ValueError(f"Unsafe or duplicate pretrained destination: {required['destination']}")
        if len(required["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in required["sha256"]):
            raise ValueError(f"Invalid SHA-256 for {required['name']}")
        destinations.add(required["destination"])
        result.append(required)
    return result


def fetch_assets(manifest: Iterable[dict[str, str]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for item in manifest:
        destination = output_dir / item["destination"]
        if destination.is_file() and sha256_file(destination) == item["sha256"]:
            downloaded.append(destination)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                request = urllib.request.Request(item["url"], headers={"User-Agent": "YOLO26-App-build"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    final_url = urlparse(response.geturl())
                    if final_url.scheme != "https" or final_url.hostname not in ALLOWED_HOSTS:
                        raise ValueError("Pretrained asset redirected to an untrusted host.")
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            if sha256_file(temporary_path) != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {item['name']}")
            os.replace(temporary_path, destination)
            downloaded.append(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST.parent)
    args = parser.parse_args()
    for asset in fetch_assets(load_manifest(args.manifest), args.output_dir):
        print(f"Verified {asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
