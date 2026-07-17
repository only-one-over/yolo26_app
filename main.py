import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# 代码核心在 code/ 文件夹下,用户更新项目只需替换 code/
sys.path.insert(0, str(Path(__file__).parent / "code"))

from yolo26_app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 App")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
