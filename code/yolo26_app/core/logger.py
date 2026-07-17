"""统一日志模块。

为整个 YOLO26 App 提供集中的日志记录能力：
- 通过 ``init_logging(workspace_root)`` 在程序启动时初始化根 logger，
  将日志按日期切割写入 ``workspace_root/logs/app.log``，保留 7 天。
- 控制台同时输出 WARNING 及以上级别，便于开发调试。
- 业务模块通过 ``get_logger(__name__)`` 获取配置好的 logger 实例，
  无需关心 handler 的添加与去重。
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# 统一日志格式
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 模块级初始化标志，避免重复添加 handler
_initialized: bool = False

# 初始化时记录的 workspace_root，供其他模块（如 exception_handler）复用
_workspace_root: Optional[Path] = None


def init_logging(workspace_root: Path) -> None:
    """初始化全局日志体系。

    在 ``main.py`` 启动时调用一次即可。重复调用会被 ``_initialized`` 标志拦截。

    Parameters
    ----------
    workspace_root:
        项目根目录，日志将写入 ``workspace_root / "logs" / "app.log"``。
    """
    global _initialized, _workspace_root
    if _initialized:
        return

    logs_dir = workspace_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 文件 handler：按日期切割，保留 7 天
    file_handler = TimedRotatingFileHandler(
        filename=str(logs_dir / "app.log"),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setFormatter(_FORMATTER)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # 控制台 handler：仅输出 WARNING 及以上级别
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(_FORMATTER)
    stream_handler.setLevel(logging.WARNING)
    root_logger.addHandler(stream_handler)

    _workspace_root = workspace_root
    _initialized = True

    logging.getLogger(__name__).info(
        "日志系统已初始化，日志目录：%s", logs_dir
    )


def get_logger(name: str) -> logging.Logger:
    """获取配置好的 logger 实例。

    若 ``init_logging`` 尚未调用，则返回的 logger 仍可正常使用（仅输出到
    默认的 lastResort handler），不会抛出异常。

    Parameters
    ----------
    name:
        通常传入 ``__name__``。
    """
    return logging.getLogger(name)


def get_workspace_root() -> Optional[Path]:
    """返回 ``init_logging`` 时传入的 workspace_root，便于其他模块复用。

    若尚未初始化则返回 ``None``。
    """
    return _workspace_root
