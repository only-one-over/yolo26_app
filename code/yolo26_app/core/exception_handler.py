"""全局异常处理模块。

在程序启动时安装 ``sys.excepthook`` 与 PyQt6 事件循环异常钩子，捕获
未处理异常，并在崩溃前：
1. 强制保存当前标注（``AnnotateWidget.flush_autosave``）。
2. 写入崩溃日志快照到 ``workspace/logs/crash_YYYYMMDD_HHMMSS.log``，
   含时间戳、系统信息与完整 traceback。
3. 弹出 ``QMessageBox.critical`` 友好提示，引导用户导出诊断报告后提交 Issue。

调用方式：
    from yolo26_app.core.exception_handler import install_exception_hooks
    install_exception_hooks(main_window)
"""
from __future__ import annotations

import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtWidgets import QApplication, QMessageBox

from yolo26_app.core.logger import get_logger, get_workspace_root

logger = get_logger(__name__)

# 保存原 sys.excepthook 引用，便于调试时回退
_original_excepthook = sys.excepthook

# 持有 MainWindow 引用，供 excepthook 中调用 flush_autosave
_main_window_ref: Optional[Any] = None


def _resolve_logs_dir() -> Path:
    """返回崩溃日志应写入的 logs 目录。

    优先使用 ``init_logging`` 时记录的 workspace_root；若未初始化则回退
    到项目根目录（main.py 所在目录的上一级）。
    """
    workspace_root = get_workspace_root()
    if workspace_root is not None:
        logs_dir = workspace_root / "logs"
    else:
        # 回退：基于当前文件位置推断项目根（code/yolo26_app/core -> 项目根）
        workspace_root = Path(__file__).resolve().parent.parent.parent.parent
        logs_dir = workspace_root / "logs"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return logs_dir


def _flush_autosave_safely() -> None:
    """安全调用 MainWindow.annotate_widget.flush_autosave()。

    若 annotate_widget 为 None 或调用失败，记录 warning 日志但不抛出异常，
    避免在异常处理流程中引发二次崩溃。
    """
    if _main_window_ref is None:
        logger.warning("崩溃时 main_window_ref 为 None，跳过标注自动保存")
        return

    annotate_widget = getattr(_main_window_ref, "annotate_widget", None)
    if annotate_widget is None:
        logger.warning("崩溃时 annotate_widget 为 None，跳过标注自动保存")
        return

    try:
        annotate_widget.flush_autosave()
        logger.info("崩溃前已成功调用 flush_autosave 保存当前标注")
    except Exception as exc:  # noqa: BLE001 - 异常处理流程中需吞掉二次异常
        logger.warning("崩溃前调用 flush_autosave 失败：%s", exc, exc_info=True)


def _write_crash_log(exc_type: type, exc_value: BaseException, exc_tb: Any) -> Optional[Path]:
    """写入崩溃日志快照到 workspace/logs/crash_YYYYMMDD_HHMMSS.log。

    Returns
    -------
    Optional[Path]
        成功写入则返回崩溃日志路径，失败则返回 ``None``。
    """
    logs_dir = _resolve_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crash_log_path = logs_dir / f"crash_{timestamp}.log"

    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = "".join(tb_lines)

    content_lines = [
        "=" * 60,
        f"YOLO26 App 崩溃日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"时间戳: {datetime.now().isoformat()}",
        f"操作系统: {platform.platform()}",
        f"Python 版本: {platform.python_version()}",
        "",
        "Traceback:",
        tb_text,
    ]

    try:
        crash_log_path.write_text(
            "\n".join(content_lines), encoding="utf-8"
        )
        return crash_log_path
    except OSError as exc:
        logger.error(
            "写入崩溃日志失败：%s（目标路径：%s）", exc, crash_log_path
        )
        return None


def _show_crash_dialog(crash_log_path: Optional[Path]) -> None:
    """弹出非阻塞的崩溃提示对话框。"""
    path_str = str(crash_log_path) if crash_log_path else "<写入失败>"
    message = (
        "程序遇到错误，已自动保存当前标注。\n"
        f"崩溃日志已写入 {path_str}。\n"
        "建议导出诊断报告后提交 Issue。"
    )
    try:
        QMessageBox.critical(
            None,
            "程序崩溃",
            message,
            QMessageBox.StandardButton.Ok,
            QMessageBox.StandardButton.Ok,
        )
    except Exception as exc:  # noqa: BLE001 - 对话框失败不应阻塞退出
        logger.error("弹出崩溃提示对话框失败：%s", exc)


def _crash_excepthook(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
    """全局未捕获异常钩子。"""
    # 1. 强制保存当前标注（try/except 包裹，避免二次异常）
    _flush_autosave_safely()

    # 2. 写入崩溃日志快照
    crash_log_path = _write_crash_log(exc_type, exc_value, exc_tb)

    # 3. 用 logger.error 记录崩溃信息
    tb_text = "".join(
        traceback.format_exception(exc_type, exc_value, exc_tb)
    )
    logger.error(
        "未捕获的异常导致程序崩溃：%s: %s\n%s",
        exc_type.__name__,
        exc_value,
        tb_text,
    )

    # 4. 弹出友好提示（非阻塞）
    _show_crash_dialog(crash_log_path)

    # 5. 用户关闭对话框后退出程序
    #    使用 sys.exit 而非 os._exit，以便 atexit/flush 等清理逻辑有机会执行；
    #    若处于无法安全退出的状态，回退到 os._exit(1)。
    try:
        sys.exit(1)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        import os
        os._exit(1)


def _install_qt_exception_hook() -> None:
    """安装 PyQt6 事件循环异常钩子。

    ``sys.excepthook`` 仅能捕获 Python 层未处理异常；Qt 事件循环中由 C++
    抛出、经 PyQt 转换的异常有时会绕过 ``sys.excepthook``。此处通过重写
    ``QApplication.notify`` 来兜底捕获 Qt 事件派发过程中的异常。
    """
    original_notify = QApplication.notify

    def _safe_notify(self: QApplication, receiver: Any, event: Any) -> bool:
        try:
            return original_notify(self, receiver, event)
        except BaseException as exc:  # noqa: BLE001
            tb = exc.__traceback__
            _crash_excepthook(type(exc), exc, tb)
            return False

    # 仅在方法未替换的情况下绑定，避免重复包装
    if getattr(QApplication.notify, "__wrapped_by_crash_hook__", False):
        return
    try:
        QApplication.notify = _safe_notify  # type: ignore[method-assign]
        setattr(QApplication.notify, "__wrapped_by_crash_hook__", True)
    except (TypeError, AttributeError):
        # 某些 PyQt6 版本不允许直接覆盖 notify，此时跳过 Qt 兜底
        logger.warning("无法重写 QApplication.notify，Qt 事件异常兜底未安装")


def install_exception_hooks(main_window_ref: Any) -> None:
    """安装全局异常钩子（Python + Qt）。

    Parameters
    ----------
    main_window_ref:
        ``MainWindow`` 实例，用于在崩溃前调用 ``annotate_widget.flush_autosave``。
    """
    global _main_window_ref, _original_excepthook

    _main_window_ref = main_window_ref

    # 保存原 sys.excepthook 引用（仅首次安装时保存，便于调试回退）
    if sys.excepthook is not _crash_excepthook:
        _original_excepthook = sys.excepthook
        sys.excepthook = _crash_excepthook

    # 安装 Qt 事件循环异常兜底
    _install_qt_exception_hook()

    logger.info(
        "全局异常钩子已安装（原 sys.excepthook 已保存为 _original_excepthook）"
    )
