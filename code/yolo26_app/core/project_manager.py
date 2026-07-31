import json
from datetime import datetime
from pathlib import Path
from typing import List

from yolo26_app.core.config import ProjectConfig, TrainConfig

RECENT_PROJECTS_DIR = Path.home() / ".yolo26_app"
RECENT_PROJECTS_FILE = RECENT_PROJECTS_DIR / "recent_projects.json"
CONFIG_FILENAME = "project_config.json"
CLASSES_FILENAME = "classes.txt"
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_INVALID_PROJECT_NAME_CHARS = set('<>:"/\\|?*')


class ProjectManager:
    @staticmethod
    def validate_project_name(name: str) -> str:
        """Validate a project directory name and return its normalized value."""
        if not isinstance(name, str):
            raise ValueError("项目名称必须是字符串")
        normalized = name.strip()
        if not normalized:
            raise ValueError("项目名称不能为空")
        if normalized != name:
            raise ValueError("项目名称不能以空格开头或结尾")
        if normalized in {".", ".."}:
            raise ValueError("项目名称不能是 . 或 ..")
        if len(normalized) > 128:
            raise ValueError("项目名称不能超过 128 个字符")
        if normalized.endswith((".", " ")):
            raise ValueError("项目名称不能以句点或空格结尾")
        if any(ord(char) < 32 or char in _INVALID_PROJECT_NAME_CHARS for char in normalized):
            raise ValueError('项目名称不能包含 <>:"/\\|?* 或控制字符')
        if Path(normalized).is_absolute():
            raise ValueError("项目名称不能是绝对路径")
        reserved_stem = normalized.split(".", 1)[0].upper()
        if reserved_stem in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"项目名称不能使用 Windows 保留名称: {reserved_stem}")
        return normalized

    @staticmethod
    def resolve_project_path(name: str, path: str) -> Path:
        """Resolve a project path while guaranteeing it remains below its root."""
        normalized = ProjectManager.validate_project_name(name)
        root = Path(path).expanduser().resolve()
        project_path = (root / normalized).resolve()
        try:
            project_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("项目路径超出允许的项目根目录") from exc
        return project_path

    @staticmethod
    def create_project(name: str, path: str) -> ProjectConfig:
        name = ProjectManager.validate_project_name(name)
        project_path = ProjectManager.resolve_project_path(name, path)
        if project_path.exists():
            raise FileExistsError(f"项目已存在: {project_path}")
        project_path.mkdir(parents=True)

        (project_path / "datasets").mkdir(exist_ok=True)
        (project_path / "models").mkdir(exist_ok=True)
        (project_path / "runs").mkdir(exist_ok=True)
        (project_path / "images").mkdir(exist_ok=True)

        classes_file = project_path / CLASSES_FILENAME
        classes_file.touch(exist_ok=True)

        now = datetime.now().isoformat()
        config = ProjectConfig(
            project_name=name,
            project_path=str(project_path),
            classes=[],
            train_config=TrainConfig(),
            created_at=now,
            last_opened=now,
        )

        config_path = project_path / CONFIG_FILENAME
        config.save(config_path)

        ProjectManager.add_recent_project(str(project_path))

        return config

    @staticmethod
    def open_project(path: str) -> ProjectConfig:
        project_path = Path(path)
        config_path = project_path / CONFIG_FILENAME

        if not config_path.exists():
            raise FileNotFoundError(f"项目配置文件不存在: {config_path}")

        config = ProjectConfig.load(config_path)
        config.last_opened = datetime.now().isoformat()
        config.save(config_path)

        ProjectManager.add_recent_project(str(project_path))

        return config

    @staticmethod
    def get_recent_projects() -> List[str]:
        if not RECENT_PROJECTS_FILE.exists():
            return []

        try:
            with open(RECENT_PROJECTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("projects", [])
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def add_recent_project(path: str) -> None:
        try:
            RECENT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            return

        projects = ProjectManager.get_recent_projects()

        if path in projects:
            projects.remove(path)
        projects.insert(0, path)

        projects = projects[:20]

        try:
            with open(RECENT_PROJECTS_FILE, "w", encoding="utf-8") as f:
                json.dump({"projects": projects}, f, indent=2, ensure_ascii=False)
        except (PermissionError, OSError):
            pass

    @staticmethod
    def get_dataset_dir(config: ProjectConfig) -> Path:
        return Path(config.project_path) / "datasets"

    @staticmethod
    def get_images_dir(config: ProjectConfig) -> Path:
        return Path(config.project_path) / "images"

    @staticmethod
    def get_models_dir(config: ProjectConfig) -> Path:
        return Path(config.project_path) / "models"

    @staticmethod
    def get_annotations_path(config: ProjectConfig) -> Path:
        return Path(config.project_path) / "annotations.json"
