"""工作区路径常量与目录初始化。

集中管理系统模型目录、用户项目目录等路径，避免散落在各模块中。
"""
from pathlib import Path


def _resolve_workspace_root() -> Path:
    """Use the checkout while developing and a user-writable directory when installed."""
    checkout_root = Path(__file__).resolve().parents[3]
    if (checkout_root / "pyproject.toml").is_file():
        return checkout_root
    return Path.home() / ".yolo26_app" / "workspace"


WORKSPACE_ROOT = _resolve_workspace_root()

# 系统模型目录（按用途分子文件夹）
SYSTEM_MODEL_DIR = WORKSPACE_ROOT / "system_model"
SYSTEM_MODEL_SUBDIRS = {
    "yolo": SYSTEM_MODEL_DIR / "yolo",                  # 训练预训练模型
    "sam2": SYSTEM_MODEL_DIR / "sam2",                  # SAM2 分割模型
    "grounding_dino": SYSTEM_MODEL_DIR / "grounding_dino",  # GroundingDINO
}

# 用户训练模型目录（存放在系统模型目录下，便于统一管理）
USER_TRAINED_MODELS_DIR = SYSTEM_MODEL_DIR / "user_trained"

# 用户项目根目录
PROJECTS_ROOT = WORKSPACE_ROOT / "my_project"

# 默认工作区间（自由空间模式时数据持久化到此目录）
DEFAULT_PROJECT_DIR = PROJECTS_ROOT / "default"

# 应用状态目录（沿用已有，~/.yolo26_app）
APP_DATA_DIR = Path.home() / ".yolo26_app"


def ensure_workspace_dirs() -> None:
    """首次运行时创建工作区目录结构。"""
    SYSTEM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for d in SYSTEM_MODEL_SUBDIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    USER_TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
