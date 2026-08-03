"""GUI application entry point for installed and source-tree runs."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from yolo26_app.core.exception_handler import install_exception_hooks
from yolo26_app.core.bundled_models import BundledModelInstallWorker, bundled_models
from yolo26_app.core.logger import init_logging
from yolo26_app.core.paths import WORKSPACE_ROOT, ensure_workspace_dirs
from yolo26_app.core.startup_metrics import StartupMetrics
from yolo26_app.ui.main_window import MainWindow


def main() -> int:
    """Start the YOLO26 desktop application."""
    metrics = StartupMetrics(WORKSPACE_ROOT)
    metrics.mark("process_entry")
    ensure_workspace_dirs()
    metrics.mark("workspace_dirs_ready")
    init_logging(WORKSPACE_ROOT)

    app = QApplication(sys.argv)
    metrics.mark("qt_application_ready")
    app.setApplicationName("YOLO26 App")
    app.setStyle("Fusion")

    window = MainWindow(metrics=metrics)
    metrics.mark("main_window_constructed")
    install_exception_hooks(window)
    window.show()
    metrics.mark("main_window_shown")
    QTimer.singleShot(0, lambda: metrics.mark("main_window_interactive"))
    if bundled_models():
        worker = BundledModelInstallWorker(None)
        window._bundled_model_install_worker = worker
        window.statusbar.showMessage("正在准备内置模型...", 5000)
        worker.completed.connect(
            lambda installed, skipped: window.statusbar.showMessage(
                f"内置模型准备完成：新增 {installed} 个，保留 {skipped} 个", 6000
            )
        )
        worker.failed.connect(lambda message: window.statusbar.showMessage(f"内置模型准备失败：{message}", 8000))
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(lambda: setattr(window, "_bundled_model_install_worker", None))
        worker.start()

    return app.exec()
