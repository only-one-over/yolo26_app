import json
from pathlib import Path
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
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QAction, QCloseEvent

from yolo26_app.core.config import ProjectConfig
from yolo26_app.core.project_manager import ProjectManager
from yolo26_app.ui.styles import DARK_STYLE
from yolo26_app.ui.annotate_widget import AnnotateWidget
from yolo26_app.ui.train_widget import TrainWidget
from yolo26_app.ui.test_widget import TestWidget
from yolo26_app.core.gpu_detector import GPUDetectWorker, load_exit_flag, save_exit_flag

APP_STATE_DIR = Path.home() / ".yolo26_app"
APP_STATE_FILE = APP_STATE_DIR / "app_state.json"


class NewProjectDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建项目")
        self.setMinimumWidth(460)
        self.setModal(True)

        layout = QFormLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("输入项目名称")
        layout.addRow("项目名称:", self.name_edit)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择项目存储路径")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setMinimumWidth(80)
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
        self._gpu_detect_worker: Optional[GPUDetectWorker] = None

        save_exit_flag(False)

        self.setWindowTitle("YOLO26 App")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 800)

        self._init_ui()
        self._init_menu()
        self._init_statusbar()
        self._apply_style()
        QTimer.singleShot(0, self._restore_app_state)
        QTimer.singleShot(0, self._detect_gpu_async)

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(80)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

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
        self.train_widget: Optional[TrainWidget] = None
        self.test_widget: Optional[TestWidget] = None
        self._widgets_created: dict = {0: True, 1: False, 2: False}

        self.stacked = QStackedWidget()
        self.stacked.addWidget(self.annotate_widget)
        main_layout.addWidget(self.stacked, 1)

        self.nav_buttons[0].setChecked(True)

    def _switch_page(self, index: int) -> None:
        if not self._widgets_created.get(index):
            self._ensure_widget(index)
        self.stacked.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def _ensure_widget(self, index: int) -> None:
        if index == 1 and self.train_widget is None:
            self.train_widget = TrainWidget()
            self.stacked.addWidget(self.train_widget)
            if self.current_project_config is not None:
                self.train_widget.set_project_config(self.current_project_config)
            self._widgets_created[1] = True
        elif index == 2 and self.test_widget is None:
            self.test_widget = TestWidget()
            self.test_widget.model_loaded.connect(self.annotate_widget.set_yolo_model)
            self.stacked.addWidget(self.test_widget)
            if self.current_project_config is not None:
                self.test_widget.set_project_config(self.current_project_config)
            self._widgets_created[2] = True

    def _set_project_config(self, config: ProjectConfig) -> None:
        self.current_project_config = config
        self.setWindowTitle(f"YOLO26 App - {config.project_name}")
        self.annotate_widget.set_project_config(config)
        if self.train_widget is not None:
            self.train_widget.set_project_config(config)
        if self.test_widget is not None:
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
        self._device_label = QLabel()
        self._device_label.setObjectName("deviceLabel")
        self._device_label.setText("⏳ 检测设备...")
        self.statusbar.addPermanentWidget(self._device_label)

    def _detect_gpu_async(self) -> None:
        exit_flag = load_exit_flag()
        if exit_flag is False:
            self._device_label.setText("🔴 CPU (安全模式)")
            self.statusbar.showMessage("上次未正常退出，已启用安全模式", 5000)
            return
        self._gpu_detect_worker = GPUDetectWorker(self)
        self._gpu_detect_worker.result_ready.connect(self._on_gpu_detected)
        self._gpu_detect_worker.finished.connect(self._gpu_detect_worker.deleteLater)
        self._gpu_detect_worker.start()

    def _on_gpu_detected(self, status: str, device_name: str) -> None:
        if status == "gpu":
            self._device_label.setText(f"🟢 GPU: {device_name}")
        elif status == "timeout":
            self._device_label.setText("🔴 CPU (检测超时)")
            self.statusbar.showMessage("GPU 检测超时，已降级为 CPU 模式", 5000)
        else:
            self._device_label.setText("🔴 CPU")

    def _apply_style(self) -> None:
        self.setStyleSheet(DARK_STYLE)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_app_state()
        save_exit_flag(True)
        event.accept()

    def _save_app_state(self) -> None:
        state = {
            "geometry": {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
            },
        }
        if self.current_project_config is not None:
            state["last_project_path"] = self.current_project_config.project_path
            annotate_state = self.annotate_widget.save_state()
            state["annotate_state"] = annotate_state
        try:
            APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(APP_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except (PermissionError, OSError):
            pass

    def _restore_app_state(self) -> None:
        if not APP_STATE_FILE.exists():
            return
        try:
            with open(APP_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        geo = state.get("geometry")
        if geo:
            self.setGeometry(geo.get("x", 100), geo.get("y", 100),
                             geo.get("width", 1280), geo.get("height", 800))
        last_project = state.get("last_project_path")
        if last_project and Path(last_project).exists():
            try:
                config = ProjectManager.open_project(last_project)
                self._set_project_config(config)
            except Exception:
                pass
        annotate_state = state.get("annotate_state")
        if annotate_state:
            self.annotate_widget.restore_state(annotate_state)

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
