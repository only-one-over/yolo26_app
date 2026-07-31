"""Tests for the reproducible Windows portable build configuration."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "build_windows_onedir.py"


def _load_build_module():
    spec = importlib.util.spec_from_file_location("build_windows_onedir", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_onedir_command_collects_runtime_and_resources(tmp_path: Path) -> None:
    module = _load_build_module()

    args, app_dir = module.build_pyinstaller_args(PROJECT_ROOT, "cpu", tmp_path)

    assert app_dir == tmp_path / "YOLO26-App-CPU"
    assert "--onedir" in args
    assert "--windowed" in args
    assert args[args.index("--name") + 1] == "YOLO26-App-CPU"
    assert "PyQt6" in args
    assert "ultralytics" in args
    assert "torch" in args
    assert "torchvision" in args
    yaml_source = str(PROJECT_ROOT / "code" / "yolo26_app" / "core" / "config_template.yaml")
    icon_source = str(PROJECT_ROOT / "code" / "yolo26_app" / "ui" / "icons")
    assert any(argument.startswith(yaml_source) for argument in args)
    assert any(argument.startswith(icon_source) for argument in args)


def test_cuda_variant_and_invalid_variant_handling(tmp_path: Path) -> None:
    module = _load_build_module()

    args, app_dir = module.build_pyinstaller_args(PROJECT_ROOT, "cuda", tmp_path)

    assert app_dir.name == "YOLO26-App-CUDA"
    assert args[args.index("--name") + 1] == "YOLO26-App-CUDA"
    try:
        module.app_name("unknown")
    except ValueError as exc:
        assert "Unsupported Windows runtime variant" in str(exc)
    else:
        raise AssertionError("Invalid variant should be rejected.")


def test_large_release_archive_is_split_with_reassembly_metadata(tmp_path: Path) -> None:
    module = _load_build_module()
    archive = tmp_path / "YOLO26-App-CUDA.zip"
    archive.write_bytes(b"0123456789")
    archive.with_suffix(".zip.sha256").write_text("checksum  YOLO26-App-CUDA.zip\n", encoding="ascii")

    assets = module.split_release_archive(archive, max_asset_size=4)

    assert [asset.name for asset in assets] == [
        "YOLO26-App-CUDA.zip.part001",
        "YOLO26-App-CUDA.zip.part002",
        "YOLO26-App-CUDA.zip.part003",
        "YOLO26-App-CUDA.zip.sha256",
        "YOLO26-App-CUDA.reassemble.ps1",
    ]
    assert b"".join(asset.read_bytes() for asset in assets[:3]) == b"0123456789"
    assert not archive.exists()
    assert "Get-FileHash" in assets[-1].read_text(encoding="utf-8")


def test_windows_release_workflow_builds_and_publishes_both_variants() -> None:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "windows-release.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    build_job = workflow["jobs"]["build"]
    variants = build_job["strategy"]["matrix"]["include"]

    assert {item["variant"] for item in variants} == {"cpu", "cuda"}
    assert {item["torch_index"] for item in variants} == {
        "https://download.pytorch.org/whl/cpu",
        "https://download.pytorch.org/whl/cu121",
    }
    assert workflow["jobs"]["release"]["needs"] == "build"
    assert workflow["jobs"]["release"]["permissions"]["contents"] == "write"
    release_files = workflow["jobs"]["release"]["steps"][-1]["with"]["files"]
    assert "release/*.zip.part*" in release_files
    assert "release/*.reassemble.ps1" in release_files
    install_step = next(
        step for step in build_job["steps"] if step.get("name") == "Install ${{ matrix.variant }} runtime"
    )
    assert "numpy==1.26.4 opencv-python==4.10.0.84 lap>=0.5.12" in install_step["run"]
