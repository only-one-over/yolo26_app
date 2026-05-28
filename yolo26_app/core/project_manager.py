import json
from datetime import datetime
from pathlib import Path
from typing import List

from yolo26_app.core.config import ProjectConfig, TrainConfig, ClassItem

RECENT_PROJECTS_DIR = Path.home() / ".yolo26_app"
RECENT_PROJECTS_FILE = RECENT_PROJECTS_DIR / "recent_projects.json"
CONFIG_FILENAME = "project_config.json"
CLASSES_FILENAME = "classes.txt"


class ProjectManager:
    @staticmethod
    def create_project(name: str, path: str) -> ProjectConfig:
        project_path = Path(path) / name
        project_path.mkdir(parents=True, exist_ok=True)

        (project_path / "datasets").mkdir(exist_ok=True)
        (project_path / "models").mkdir(exist_ok=True)
        (project_path / "runs").mkdir(exist_ok=True)

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
    def get_models_dir(config: ProjectConfig) -> Path:
        return Path(config.project_path) / "models"
