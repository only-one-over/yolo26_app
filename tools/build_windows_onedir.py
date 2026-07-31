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


VARIANTS = ("cpu", "cuda")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    }
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
                "TensorRT, SAM2, Grounding DINO, and RealSense remain optional dependencies.",
                "",
            ]
        ),
        encoding="utf-8",
    )


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
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="ascii"
    )
    return archive


def build(variant: str, output_dir: Path) -> Path:
    """Run PyInstaller and archive a completed portable distribution."""
    args, app_dir = build_pyinstaller_args(PROJECT_ROOT, variant, output_dir)
    if app_dir.exists():
        shutil.rmtree(app_dir)

    from PyInstaller.__main__ import run as pyinstaller_run

    pyinstaller_run(args)
    executable = app_dir / f"{app_name(variant)}.exe"
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller did not create the expected launcher: {executable}")
    write_distribution_files(app_dir, variant)
    return create_archive(app_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "release")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    archive = build(args.variant, args.output_dir.resolve())
    print(f"Created {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
