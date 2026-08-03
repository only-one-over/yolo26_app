"""Build a portable Windows onedir distribution for one runtime variant."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence


VARIANTS = ("cpu", "cuda", "cuda-full")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_ASSET_SIZE_MIB = 1900


def app_name(variant: str) -> str:
    """Return the user-facing directory and executable name for a variant."""
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Windows runtime variant: {variant}")
    return f"YOLO26-App-{variant.upper()}"


def build_pyinstaller_args(
    project_root: Path, variant: str, output_dir: Path
) -> tuple[list[str], Path]:
    """Create the PyInstaller command for a Windows one-folder build."""
    name = app_name(variant)
    app_dir = output_dir / name
    build_root = project_root / "build" / "pyinstaller" / variant
    separator = ";" if sys.platform.startswith("win") else ":"

    def add_data(source: Path, destination: str) -> list[str]:
        return ["--add-data", f"{source.resolve()}{separator}{destination}"]

    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--contents-directory",
        ".",
        "--name",
        name,
        "--distpath",
        str(output_dir),
        "--workpath",
        str(build_root / "work"),
        "--specpath",
        str(build_root / "spec"),
        "--paths",
        str(project_root / "code"),
        "--collect-all",
        "PyQt6",
        "--collect-all",
        "ultralytics",
        "--collect-all",
        "torch",
        "--collect-all",
        "torchvision",
        "--collect-all",
        "cv2",
        "--collect-all",
        "pyqtgraph",
        "--hidden-import",
        "yaml",
    ]
    args += add_data(
        project_root / "code" / "yolo26_app" / "core" / "config_template.yaml",
        "yolo26_app/core",
    )
    args += add_data(project_root / "code" / "yolo26_app" / "ui" / "icons", "yolo26_app/ui/icons")
    if variant == "cuda-full":
        args += [
            "--collect-all", "sam2",
            "--collect-all", "sam2_configs",
            "--collect-all", "hydra",
            "--collect-all", "omegaconf",
            "--collect-all", "iopath",
        ]
        args += add_data(project_root / "assets" / "pretrained", "pretrained")
    args.append(str(project_root / "main.py"))
    return args, app_dir


def package_version(package: str) -> str | None:
    """Return an installed package version without making the build fail."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def write_distribution_files(app_dir: Path, variant: str) -> None:
    """Place runtime metadata and basic user-facing files in the app folder."""
    torch_cuda: str | None = None
    try:
        import torch

        torch_cuda = torch.version.cuda
    except ImportError:
        pass

    metadata = {
        "application": "YOLO26 App",
        "variant": variant,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyinstaller": package_version("pyinstaller"),
        "torch": package_version("torch"),
        "torch_cuda": torch_cuda,
        "ultralytics": package_version("ultralytics"),
        "bundled_models": [],
    }
    manifest = app_dir / "pretrained" / "manifest.json"
    if manifest.is_file():
        try:
            metadata["bundled_models"] = [item["name"] for item in json.loads(manifest.read_text("utf-8"))["models"]]
        except (KeyError, OSError, ValueError, TypeError):
            pass
    (app_dir / "build-info.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(PROJECT_ROOT / "LICENSE", app_dir / "LICENSE")
    (app_dir / "README.txt").write_text(
        "\n".join(
            [
                "YOLO26 App portable Windows distribution",
                "",
                f"Runtime variant: {variant.upper()}",
                f"Start the application with {app_name(variant)}.exe.",
                "",
                "CPU: works on Windows x64 without an NVIDIA GPU.",
                "CUDA: requires a compatible NVIDIA GPU and driver; CUDA-enabled PyTorch is bundled.",
                "",
                "User projects, annotations, models, and logs are stored in:",
                "%USERPROFILE%\\.yolo26_app\\workspace",
                "",
                "TensorRT, Grounding DINO, and RealSense remain optional dependencies.",
                "CUDA-FULL additionally includes SAM2 Tiny and YOLO26 n/s pretrained models.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without loading a large archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_archive(app_dir: Path) -> Path:
    """Create a ZIP whose top-level directory is the onedir application folder."""
    archive_base = app_dir.parent / app_dir.name
    archive = Path(
        shutil.make_archive(
            str(archive_base),
            "zip",
            root_dir=app_dir.parent,
            base_dir=app_dir.name,
        )
    )
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii"
    )
    return archive


def write_reassembly_script(archive: Path) -> Path:
    """Write a PowerShell helper that joins split Release assets and verifies them."""
    script = archive.with_suffix(".reassemble.ps1")
    script.write_text(
        "\n".join(
            [
                "$ErrorActionPreference = 'Stop'",
                f"$archive = Join-Path $PSScriptRoot '{archive.name}'",
                f"$parts = Get-ChildItem -LiteralPath $PSScriptRoot -Filter '{archive.name}.part*' | Sort-Object Name",
                "if ($parts.Count -eq 0) { throw 'No archive parts were found.' }",
                "$output = [System.IO.File]::Open($archive, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)",
                "try {",
                "  foreach ($part in $parts) {",
                "    $input = [System.IO.File]::OpenRead($part.FullName)",
                "    try { $input.CopyTo($output) } finally { $input.Dispose() }",
                "  }",
                "} finally { $output.Dispose() }",
                f"$expected = (Get-Content -LiteralPath (Join-Path $PSScriptRoot '{archive.name}.sha256')).Split()[0]",
                "$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()",
                "if ($actual -ne $expected) { throw 'Archive checksum verification failed.' }",
                "Write-Host \"Created and verified $archive\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return script


def split_release_archive(archive: Path, max_asset_size: int) -> list[Path]:
    """Split an oversized Release ZIP into GitHub-compatible sequential parts."""
    if archive.stat().st_size <= max_asset_size:
        return [archive, archive.with_suffix(archive.suffix + ".sha256")]

    parts: list[Path] = []
    with archive.open("rb") as source:
        index = 1
        while chunk := source.read(max_asset_size):
            part = archive.with_name(f"{archive.name}.part{index:03d}")
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    archive.unlink()
    parts.append(archive.with_suffix(archive.suffix + ".sha256"))
    parts.append(write_reassembly_script(archive))
    return parts


def build(variant: str, output_dir: Path, max_asset_size: int) -> list[Path]:
    """Run PyInstaller and archive a completed portable distribution."""
    args, app_dir = build_pyinstaller_args(PROJECT_ROOT, variant, output_dir)
    if variant == "cuda-full":
        manifest = PROJECT_ROOT / "assets" / "pretrained" / "manifest.json"
        try:
            assets = json.loads(manifest.read_text(encoding="utf-8"))["models"]
            missing = [item["destination"] for item in assets if not (manifest.parent / item["destination"]).is_file()]
        except (KeyError, OSError, ValueError, TypeError) as exc:
            raise RuntimeError("CUDA-Full pretrained asset manifest is invalid.") from exc
        if missing:
            raise RuntimeError(f"CUDA-Full pretrained assets are missing: {', '.join(missing)}")
    if app_dir.exists():
        shutil.rmtree(app_dir)

    from PyInstaller.__main__ import run as pyinstaller_run

    pyinstaller_run(args)
    executable = app_dir / f"{app_name(variant)}.exe"
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller did not create the expected launcher: {executable}")
    write_distribution_files(app_dir, variant)
    return split_release_archive(create_archive(app_dir), max_asset_size)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "release")
    parser.add_argument(
        "--max-asset-size-mib",
        type=int,
        default=DEFAULT_MAX_ASSET_SIZE_MIB,
        help="Maximum size of a single GitHub Release asset before ZIP splitting.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    assets = build(
        args.variant,
        args.output_dir.resolve(),
        args.max_asset_size_mib * 1024 * 1024,
    )
    for asset in assets:
        print(f"Created {asset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
