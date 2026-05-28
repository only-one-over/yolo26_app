from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QLabel,
    QFrame,
    QMenuBar,
    QMenu,
    QStatusBar,
    QDialog,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction

from yolo26_app.core.config import ProjectConfig
from yolo26_app.core.project_manager import ProjectManager
from yolo26_app.ui.styles import DARK_STYLE
from yolo26_app.ui.annotate_widget import AnnotateWidget
from yolo26_app.ui.train_widget import TrainWidget
from yolo26_app.ui.test_widget import TestWidget


class NewProjectDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(460)
        self.setModal(True)

        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入项目名称")
        layout.addRow("项目名称:", self.name_edit)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择项目存储路径")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self._browse_path)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_btn)
        layout.addRow("项目路径:", path_row)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def _browse_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目路径")
        if path:
            self.path_edit.setText(path)

    def get_project_info(self) -> Tuple[str, str]:
        return self.name_edit.text().strip(), self.path_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.current_project_config: Optional[ProjectConfig] = None

        self.setWindowTitle("YOLO26 App")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 800)

        self._init_ui()
        self._init_menu()
        self._init_statusbar()
        self._apply_style()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(80)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 8, 4, 8)
        sidebar_layout.setSpacing(4)

        self.nav_buttons: List[QPushButton] = []
        nav_items: List[Tuple[str, str, int]] = [
            ("🏷️", "标注", 0),
            ("🏋️", "训练", 1),
            ("🔍", "测试", 2),
        ]

        for icon_text, label_text, index in nav_items:
            btn = QPushButton(f"{icon_text}\n{label_text}")
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda checked, idx=index: self._switch_page(idx))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        self.annotate_widget = AnnotateWidget()
        self.train_widget = TrainWidget()
        self.test_widget = TestWidget()

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.annotate_widget)
        self.stacked.addWidget(self.train_widget)
        self.stacked.addWidget(self.test_widget)
        main_layout.addWidget(self.stacked, 1)

        self.nav_buttons[0].setChecked(True)

    def _switch_page(self, index: int) -> None:
        self.stacked.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def _set_project_config(self, config: ProjectConfig) -> None:
        self.current_project_config = config
        self.setWindowTitle(f"YOLO26 App - {config.project_name}")
        self.annotate_widget.set_project_config(config)
        self.train_widget.set_project_config(config)
        self.test_widget.set_project_config(config)

    def _init_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")

        new_action = QAction("新建项目(&N)", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_project)
        file_menu.addAction(new_action)

        open_action = QAction("打开项目(&O)", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_project)
        file_menu.addAction(open_action)

        self.recent_menu = QMenu("最近项目(&R)", self)
        file_menu.addMenu(self.recent_menu)
        self._refresh_recent_projects()

        file_menu.addSeparator()

        exit_action = QAction("退出(&Q)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _init_statusbar(self) -> None:
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪")

    def _apply_style(self) -> None:
        self.setStyleSheet(DARK_STYLE)

    def _new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name, path = dialog.get_project_info()
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        if not path:
            QMessageBox.warning(self, "提示", "请选择项目路径")
            return

        try:
            config = ProjectManager.create_project(name, path)
            self._set_project_config(config)
            self.statusbar.showMessage(f"已创建项目: {config.project_name}", 5000)
            self._refresh_recent_projects()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建项目失败:\n{e}")

    def _open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择项目目录")
        if not path:
            return

        try:
            config = ProjectManager.open_project(path)
            self._set_project_config(config)
            self.statusbar.showMessage(f"已打开项目: {config.project_name}", 5000)
            self._refresh_recent_projects()
        except FileNotFoundError as e:
            QMessageBox.critical(self, "错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开项目失败:\n{e}")

    def _open_recent_project(self, path: str) -> None:
        try:
            config = ProjectManager.open_project(path)
            self._set_project_config(config)
            self.statusbar.showMessage(f"已打开项目: {config.project_name}", 5000)
            self._refresh_recent_projects()
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", f"项目不存在: {path}")
            self._refresh_recent_projects()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开项目失败:\n{e}")

    def _refresh_recent_projects(self) -> None:
        self.recent_menu.clear()
        projects = ProjectManager.get_recent_projects()

        if not projects:
            empty_action = QAction("（无最近项目）", self)
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)
            return

        for proj_path in projects:
            action = QAction(proj_path, self)
            action.triggered.connect(lambda checked, p=proj_path: self._open_recent_project(p))
            self.recent_menu.addAction(action)
