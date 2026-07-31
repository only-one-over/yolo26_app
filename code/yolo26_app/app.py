"""GUI application entry point for installed and source-tree runs."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from yolo26_app.core.exception_handler import install_exception_hooks
from yolo26_app.core.logger import init_logging
from yolo26_app.core.paths import WORKSPACE_ROOT, ensure_workspace_dirs
from yolo26_app.ui.main_window import MainWindow


def main() -> int:
    """Start the YOLO26 desktop application."""
    ensure_workspace_dirs()
    init_logging(WORKSPACE_ROOT)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 App")
    app.setStyle("Fusion")

    window = MainWindow()
    install_exception_hooks(window)
    window.show()

    return app.exec()
