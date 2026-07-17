import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

# 代码核心在 code/ 文件夹下,用户更新项目只需替换 code/
sys.path.insert(0, str(Path(__file__).parent / "code"))

from yolo26_app.ui.main_window import MainWindow
from yolo26_app.core.logger import init_logging
from yolo26_app.core.exception_handler import install_exception_hooks


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 App")
    app.setStyle("Fusion")

    # 初始化统一日志体系（logs 子目录会自动创建）
    init_logging(Path(__file__).parent)

    window = MainWindow()
    # 安装全局异常钩子，崩溃前自动保存标注并写入崩溃日志
    install_exception_hooks(window)

    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
