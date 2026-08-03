"""GUI application entry point for installed and source-tree runs."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from yolo26_app.core.exception_handler import install_exception_hooks
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

    return app.exec()
