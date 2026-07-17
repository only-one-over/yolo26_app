"""工作区间管理器:扫描、创建、校验工作区间。

工作区间是 PROJECTS_ROOT 下的子文件夹,含 project_config.json 的为合法工作区间。
"""
import re
from pathlib import Path
from typing import List

from yolo26_app.core.config import ProjectConfig
from yolo26_app.core.paths import PROJECTS_ROOT
from yolo26_app.core.project_manager import ProjectManager, CONFIG_FILENAME


# Windows 文件名非法字符
_ILLEGAL_CHARS = re.compile(r'[\\/:*?"<>|]')


class WorkspaceManager:
    @staticmethod
    def list_workspaces() -> List[str]:
        """扫描 PROJECTS_ROOT 下所有合法工作区间(含 project_config.json 的子文件夹)。

        返回按名称排序的工作区间名称列表。
        PROJECTS_ROOT 不存在时返回空列表,不抛异常。
        """
        if not PROJECTS_ROOT.exists():
            return []
        names = []
        for child in PROJECTS_ROOT.iterdir():
            if child.is_dir() and (child / CONFIG_FILENAME).exists():
                names.append(child.name)
        return sorted(names)

    @staticmethod
    def is_valid_workspace(path: str) -> bool:
        """检查路径下是否含 project_config.json(合法工作区间)。"""
        return (Path(path) / CONFIG_FILENAME).exists()

    @staticmethod
    def create_workspace(name: str) -> ProjectConfig:
        """在 PROJECTS_ROOT 下创建新工作区间。

        校验名称:
        - 非空(strip 后)
        - 无非法字符 \\ / : * ? " < > |
        - 不与现有文件夹重名

        校验失败时抛 ValueError(含具体原因)。
        校验通过后调用 ProjectManager.create_project(name, str(PROJECTS_ROOT)) 创建。
        返回创建的 ProjectConfig。
        """
        name = name.strip()
        if not name:
            raise ValueError("工作区间名称不能为空")
        if _ILLEGAL_CHARS.search(name):
            raise ValueError('名称包含非法字符 \\ / : * ? " < > |')
        target = PROJECTS_ROOT / name
        if target.exists():
            raise ValueError(f"工作区间 '{name}' 已存在,请使用其他名称")
        return ProjectManager.create_project(name, str(PROJECTS_ROOT))

    @staticmethod
    def get_workspace_path(name: str) -> Path:
        """返回工作区间的完整路径。"""
        return PROJECTS_ROOT / name
