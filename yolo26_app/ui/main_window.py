import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

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
from yolo26_app.core.persistence import write_json_atomic

if TYPE_CHECKING:
    from yolo26_app.core.gpu_detector import GPUDetectWorker
    from yolo26_app.ui.annotate_widget import AnnotateWidget
    from yolo26_app.ui.train_widget import TrainWidget
    from yolo26_app.ui.test_widget import TestWidget

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

        # 项目名称：默认自动编号 project1, project2, ...
        self.name_edit = QLineEdit()
        from yolo26_app.core.paths import PROJECTS_ROOT, ensure_workspace_dirs
        ensure_workspace_dirs()
        default_name = self._generate_default_name(PROJECTS_ROOT)
        self.name_edit.setText(default_name)
        layout.addRow("项目名称:", self.name_edit)

        # 项目路径：只读显示 my_project 根目录
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setText(str(PROJECTS_ROOT))
        self.path_edit.setReadOnly(True)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setMinimumWidth(80)
        self.browse_btn.setVisible(False)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_btn)
        layout.addRow("项目路径:", path_row)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    @staticmethod
    def _generate_default_name(root: Path) -> str:
        """生成不冲突的默认项目名 project1, project2, ..."""
        i = 1
        while (root / f"project{i}").exists():
            i += 1
        return f"project{i}"

    def get_project_name(self) -> str:
        return self.name_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # 启动时确保工作区目录结构存在
        from yolo26_app.core.paths import ensure_workspace_dirs
        ensure_workspace_dirs()
        self.current_project_config: Optional[ProjectConfig] = None
        self._gpu_detect_worker: Optional["GPUDetectWorker"] = None
        self.annotate_widget: Optional["AnnotateWidget"] = None
        self.train_widget: Optional["TrainWidget"] = None
        self.test_widget: Optional["TestWidget"] = None
        self._page_widgets: dict[int, QWidget] = {}
        self._requested_page_index = 0

        self.setWindowTitle("YOLO26 App")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 800)

        self._init_ui()
        self._init_menu()
        self._init_statusbar()
        self._apply_style()
        self._recovery_save_timer = QTimer(self)
        self._recovery_save_timer.setSingleShot(True)
        self._recovery_save_timer.setInterval(0)
        self._recovery_save_timer.timeout.connect(self._save_app_state)
        QTimer.singleShot(25, self._finish_startup)

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

        self.stacked = QStackedWidget()
        self._startup_placeholder = QLabel("正在加载标注工作区...")
        self._startup_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stacked.addWidget(self._startup_placeholder)
        main_layout.addWidget(self.stacked, 1)

        self.nav_buttons[0].setChecked(True)

    def _switch_page(self, index: int) -> None:
        self._requested_page_index = index
        page = self._ensure_widget(index)
        if page is not None:
            self.stacked.setCurrentWidget(page)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        if hasattr(self, "_recovery_save_timer"):
            self._schedule_recovery_save()

    def _ensure_widget(self, index: int) -> Optional[QWidget]:
        existing = self._page_widgets.get(index)
        if existing is not None:
            return existing

        page: Optional[QWidget] = None
        if index == 0 and self.annotate_widget is None:
            from yolo26_app.ui.annotate_widget import AnnotateWidget

            self.annotate_widget = AnnotateWidget()
            self.annotate_widget.state_changed.connect(self._on_annotation_state_changed)
            page = self.annotate_widget
            if self.test_widget is not None:
                self.test_widget.model_loaded.connect(self.annotate_widget.set_yolo_model)
        elif index == 1 and self.train_widget is None:
            from yolo26_app.ui.train_widget import TrainWidget

            self.train_widget = TrainWidget()
            page = self.train_widget
            if self.current_project_config is not None:
                self.train_widget.set_project_config(self.current_project_config)
        elif index == 2 and self.test_widget is None:
            from yolo26_app.ui.test_widget import TestWidget

            self.test_widget = TestWidget()
            page = self.test_widget
            if self.annotate_widget is not None:
                self.test_widget.model_loaded.connect(self.annotate_widget.set_yolo_model)
            if self.current_project_config is not None:
                self.test_widget.set_project_config(self.current_project_config)
        if page is None:
            return None

        self._page_widgets[index] = page
        self.stacked.addWidget(page)
        if index == 0 and self._startup_placeholder is not None:
            self.stacked.removeWidget(self._startup_placeholder)
            self._startup_placeholder.deleteLater()
            self._startup_placeholder = None
        return page

    def _finish_startup(self) -> None:
        self._ensure_widget(0)
        self._restore_app_state()
        self._switch_page(self._requested_page_index)
        self._detect_gpu_async()

    def _set_project_config(self, config: ProjectConfig) -> None:
        if self.annotate_widget is not None and self.current_project_config is not None:
            self.annotate_widget.flush_autosave()
        self.current_project_config = config
        self.setWindowTitle(f"YOLO26 App - {config.project_name}")
        annotate_widget = self._ensure_widget(0)
        if annotate_widget is not None:
            self.annotate_widget.set_project_config(config)
        if self.train_widget is not None:
            self.train_widget.set_project_config(config)
        if self.test_widget is not None:
            self.test_widget.set_project_config(config)
        self._schedule_recovery_save()

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
        from yolo26_app.core.gpu_detector import GPUDetectWorker, load_exit_flag, save_exit_flag

        exit_flag = load_exit_flag()
        save_exit_flag(False)
        timeout = 8.0 if not exit_flag else 10.0
        if exit_flag is False:
            self._device_label.setText("⚠️ 上次未正常退出，缩短超时检测 GPU...")
            self.statusbar.showMessage("上次未正常退出，GPU 检测使用缩短超时 (8s)", 5000)
        self._gpu_detect_worker = GPUDetectWorker(self, timeout=timeout)
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
        # 检查是否有后台任务正在运行
        if self._has_running_tasks():
            reply = QMessageBox.question(
                self, "确认退出",
                "有任务正在运行中，确定要退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._stop_all_tasks()

        if self.annotate_widget is not None:
            self.annotate_widget.flush_autosave()
        self._recovery_save_timer.stop()
        self._save_app_state()
        from yolo26_app.core.gpu_detector import save_exit_flag

        save_exit_flag(True)
        event.accept()

    def _has_running_tasks(self) -> bool:
        """检查是否有后台任务正在运行"""
        # 检查训练线程
        if hasattr(self, 'train_widget') and self.train_widget is not None:
            if self.train_widget._trainer and self.train_widget._trainer.isRunning():
                return True
        # 检查摄像头/RealSense
        if hasattr(self, 'test_widget') and self.test_widget is not None:
            if hasattr(self.test_widget, 'timer') and self.test_widget.timer.isActive():
                return True
            if hasattr(self.test_widget, 'rs_camera') and self.test_widget.rs_camera.running:
                return True
        return False

    def _stop_all_tasks(self) -> None:
        """停止所有后台任务"""
        if hasattr(self, 'train_widget') and self.train_widget is not None:
            if self.train_widget._trainer and self.train_widget._trainer.isRunning():
                self.train_widget._trainer.stop()
                self.train_widget._trainer.wait(3000)
        if hasattr(self, 'test_widget') and self.test_widget is not None:
            if hasattr(self.test_widget, '_on_stop'):
                self.test_widget._on_stop()

    def _save_app_state(self) -> None:
        state = {
            "last_exit_normal": False,
            "geometry": {
                "x": self.x(),
                "y": self.y(),
                "width": self.width(),
                "height": self.height(),
            },
            "active_page": self._requested_page_index,
        }
        if self.current_project_config is not None:
            state["last_project_path"] = self.current_project_config.project_path
        elif self.annotate_widget is not None:
            state["annotate_state"] = self.annotate_widget.save_state()
        try:
            write_json_atomic(APP_STATE_FILE, state)
        except (PermissionError, OSError):
            self.statusbar.showMessage("自动恢复数据保存失败，请检查磁盘写入权限", 5000)

    def _schedule_recovery_save(self) -> None:
        self._recovery_save_timer.start()

    def _on_annotation_state_changed(self) -> None:
        if self.current_project_config is None:
            self._schedule_recovery_save()

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
        if annotate_state and self.current_project_config is None and self.annotate_widget is not None:
            self.annotate_widget.restore_state(annotate_state)
        active_page = state.get("active_page", 0)
        if isinstance(active_page, int) and active_page in (0, 1, 2):
            self._requested_page_index = active_page

    def _new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name = dialog.get_project_name()
        if not name:
            QMessageBox.warning(self, "提示", "请输入项目名称")
            return
        from yolo26_app.core.paths import PROJECTS_ROOT
        path = str(PROJECTS_ROOT)

        try:
            config = ProjectManager.create_project(name, path)
            self._set_project_config(config)
            self.statusbar.showMessage(f"已创建项目: {config.project_name}", 5000)
            self._refresh_recent_projects()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建项目失败:\n{e}")

    def _open_project(self) -> None:
        from yolo26_app.core.paths import PROJECTS_ROOT
        path = QFileDialog.getExistingDirectory(self, "选择项目目录", str(PROJECTS_ROOT))
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
