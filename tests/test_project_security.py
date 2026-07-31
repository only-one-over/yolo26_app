from pathlib import Path
from unittest.mock import patch

import pytest

from yolo26_app.core.project_manager import ProjectManager


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        ".",
        "..",
        "../escape",
        r"..\escape",
        "nested/name",
        r"nested\name",
        "project.",
        "project ",
        " project",
        "CON",
        "con.txt",
        "LPT9",
        "bad?name",
        "bad\x00name",
    ],
)
def test_project_name_rejects_unsafe_directory_names(name: str):
    with pytest.raises(ValueError):
        ProjectManager.validate_project_name(name)


def test_project_path_is_resolved_below_selected_root(tmp_path: Path):
    project_path = ProjectManager.resolve_project_path("安全项目", str(tmp_path))

    assert project_path == (tmp_path / "安全项目").resolve()
    assert project_path.parent == tmp_path.resolve()


def test_create_project_refuses_to_reuse_existing_directory(tmp_path: Path):
    (tmp_path / "existing").mkdir()

    with pytest.raises(FileExistsError):
        ProjectManager.create_project("existing", str(tmp_path))


def test_create_project_builds_expected_structure(tmp_path: Path):
    with patch.object(ProjectManager, "add_recent_project"):
        config = ProjectManager.create_project("project1", str(tmp_path))

    project_path = tmp_path / "project1"
    assert config.project_path == str(project_path.resolve())
    assert (project_path / "project_config.json").is_file()
    assert (project_path / "classes.txt").is_file()
    assert (project_path / "images").is_dir()
