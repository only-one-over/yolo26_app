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
    QComboBox,
    QInputDialog,
)
from PyQt6.QtCore import Qt, QSize, QTimer, QPoint
from PyQt6.QtGui import QAction, QCloseEvent, QIcon, QGuiApplication

from yolo26_app.core.config import ProjectConfig
from yolo26_app.core.project_manager import ProjectManager
from yolo26_app.ui.styles import DARK_STYLE, get_style
from yolo26_app.core.persistence import write_json_atomic

if TYPE_CHECKING:
    from yolo26_app.core.gpu_detector import GPUDetectWorker
    from yolo26_app.ui.annotation import AnnotateWidget
    from yolo26_app.ui.training import TrainWidget
    from yolo26_app.ui.inference import TestWidget

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
        self._last_successful_workspace = ""
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

    def _load_nav_icon(self, name: str) -> QIcon:
        """Load a navigation icon from the icons directory."""
        import os
        icon_path = os.path.join(os.path.dirname(__file__), "icons", f"{name}.svg")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QIcon()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 工作区间工具栏 (topBar)
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(48)
        toolbar_layout = QHBoxLayout(top_bar)
        toolbar_layout.setContentsMargins(8, 0, 8, 0)
        toolbar_layout.setSpacing(8)

        toolbar_layout.addWidget(QLabel("工作区间:"))
        self.workspace_combo = QComboBox()
        self.workspace_combo.setPlaceholderText("请选择工作区间")
        self.workspace_combo.setCurrentIndex(-1)
        self.workspace_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        toolbar_layout.addWidget(self.workspace_combo)

        self._new_btn = QPushButton()
        self._new_btn.setObjectName("iconButton")
        self._new_btn.setIcon(self._load_nav_icon("action-new"))
        self._new_btn.setToolTip("新建")
        self._new_btn.setFixedWidth(80)
        toolbar_layout.addWidget(self._new_btn)

        self._refresh_btn = QPushButton()
        self._refresh_btn.setObjectName("iconButton")
        self._refresh_btn.setIcon(self._load_nav_icon("action-refresh"))
        self._refresh_btn.setToolTip("刷新")
        self._refresh_btn.setFixedWidth(80)
        toolbar_layout.addWidget(self._refresh_btn)

        outer_layout.addWidget(top_bar)

        # 下方:原 sidebar + stacked 布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(0)

        self._sidebar = QFrame()
        self._sidebar.setObjectName("sidebar")
        self._sidebar.setFixedWidth(64)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        self.nav_buttons: List[QPushButton] = []
        nav_items: List[Tuple[str, str, int]] = [
            ("nav-annotate", "标注", 0),
            ("nav-train", "训练", 1),
            ("nav-test", "测试", 2),
        ]

        for icon_name, tooltip, index in nav_items:
            btn = QPushButton()
            btn.setObjectName("navButton")
            btn.setIcon(self._load_nav_icon(icon_name))
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedHeight(56)
            btn.setFixedWidth(48)
            btn.clicked.connect(lambda checked, idx=index: self._switch_page(idx))
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        sidebar_layout.addStretch()
        self._settings_btn = QPushButton()
        self._settings_btn.setObjectName("navButton")
        self._settings_btn.setIcon(self._load_nav_icon("nav-settings"))
        self._settings_btn.setToolTip("设置")
        self._settings_btn.setFixedHeight(56)
        self._settings_btn.setFixedWidth(48)
        sidebar_layout.addWidget(self._settings_btn)
        main_layout.addWidget(self._sidebar)

        self.stacked = QStackedWidget()
        self._startup_placeholder = QLabel("正在加载标注工作区...")
        self._startup_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stacked.addWidget(self._startup_placeholder)
        main_layout.addWidget(self.stacked, 1)

        outer_layout.addLayout(main_layout, 1)

        self.nav_buttons[0].setChecked(True)

        # 连接工作区间工具栏信号
        self.workspace_combo.currentIndexChanged.connect(self._on_workspace_changed)
        self._new_btn.clicked.connect(self._on_new_workspace)
        self._refresh_btn.clicked.connect(self._on_refresh_workspace)

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
            from yolo26_app.ui.annotation import AnnotateWidget

            self.annotate_widget = AnnotateWidget()
            self.annotate_widget.state_changed.connect(self._on_annotation_state_changed)
            page = self.annotate_widget
            if self.test_widget is not None:
                self.test_widget.model_loaded.connect(self.annotate_widget.set_yolo_model)
        elif index == 1 and self.train_widget is None:
            from yolo26_app.ui.training import TrainWidget

            self.train_widget = TrainWidget()
            page = self.train_widget
            if self.current_project_config is not None:
                self.train_widget.set_project_config(self.current_project_config)
        elif index == 2 and self.test_widget is None:
            from yolo26_app.ui.inference import TestWidget

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
        # 先填充工作区间列表,这样 _restore_app_state 内部调用 _set_project_config
        # 时才能在 ComboBox 中找到对应项并选中
        self._refresh_workspace_combo()
        self._ensure_widget(0)
        self._restore_app_state()
        self._switch_page(self._requested_page_index)
        self._detect_gpu_async()

    def _set_project_config(self, config: ProjectConfig) -> None:
        if self.annotate_widget is not None and self.current_project_config is not None:
            try:
                self.annotate_widget.flush_autosave()
            except Exception:
                pass
        self.current_project_config = config
        self.setWindowTitle(f"YOLO26 App - {config.project_name}")
        annotate_widget = self._ensure_widget(0)
        if annotate_widget is not None:
            try:
                self.annotate_widget.set_project_config(config)
            except Exception:
                pass
        if self.train_widget is not None:
            try:
                self.train_widget.set_project_config(config)
            except Exception:
                pass
        if self.test_widget is not None:
            try:
                self.test_widget.set_project_config(config)
            except Exception:
                pass
        self._schedule_recovery_save()
        # 同步工作区间 ComboBox 选中项(避免触发 _on_workspace_changed 递归)
        if hasattr(self, 'workspace_combo'):
            i = self.workspace_combo.findText(config.project_name)
            if i < 0:
                # ComboBox 中没有,先刷新列表
                self._refresh_workspace_combo()
                i = self.workspace_combo.findText(config.project_name)
            if i >= 0:
                self.workspace_combo.blockSignals(True)
                self.workspace_combo.setCurrentIndex(i)
                self.workspace_combo.blockSignals(False)

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
        self._project_label = QLabel("项目: -")
        self._project_label.setObjectName("projectLabel")
        self.statusbar.addPermanentWidget(self._project_label)

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
        self._gpu_detect_worker.finished.connect(lambda: setattr(self, '_gpu_detect_worker', None))
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
        self.setStyleSheet(get_style("dark"))

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
            # 停止标注页面的后台线程，避免 QThread 在 widget 销毁时仍在运行
            self.annotate_widget.stop_background_threads()
        # 停止 GPU 检测线程，避免 QThread 在窗口销毁时仍在运行
        if self._gpu_detect_worker is not None:
            try:
                if self._gpu_detect_worker.isRunning():
                    self._gpu_detect_worker.quit()
                    self._gpu_detect_worker.wait(3000)
            except RuntimeError:
                pass
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
            if hasattr(self.test_widget, '_inference_worker') and self.test_widget._inference_worker is not None:
                if self.test_widget._inference_worker.isRunning():
                    return True
        return False

    def _stop_all_tasks(self) -> None:
        """停止所有后台任务"""
        if hasattr(self, 'train_widget') and self.train_widget is not None:
            if self.train_widget._trainer and self.train_widget._trainer.isRunning():
                self.train_widget._trainer.stop()
                self.train_widget._trainer.wait(30000)
                if self.train_widget._trainer.isRunning():
                    QMessageBox.warning(self, "警告", "训练仍在运行,强制退出可能丢失数据")
        if hasattr(self, 'test_widget') and self.test_widget is not None:
            if hasattr(self.test_widget, '_on_stop'):
                self.test_widget._on_stop()

    def _save_app_state(self) -> None:
        is_maximized = self.isMaximized()
        is_fullscreen = bool(self.windowState() & Qt.WindowState.WindowFullScreen)
        if is_maximized or is_fullscreen:
            geo = self.normalGeometry()
            window_state = "maximized" if is_maximized else "fullscreen"
        else:
            geo = self.geometry()
            window_state = "normal"
        state = {
            "last_exit_normal": False,
            "geometry": {
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
            },
            "window_state": window_state,
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
            x = geo.get("x", 100)
            y = geo.get("y", 100)
            width = geo.get("width", 1280)
            height = geo.get("height", 800)

            # 获取当前屏幕可用几何
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                # 约束宽高
                width = min(width, available.width())
                height = min(height, available.height())
                width = max(width, self.minimumWidth())
                height = max(height, self.minimumHeight())
                # 检查坐标是否在任何屏幕内
                if QGuiApplication.screenAt(QPoint(x, y)) is None:
                    # 坐标脱离所有屏幕,回退到主屏
                    x = available.x() + 100
                    y = available.y() + 100
                else:
                    # 约束 x 确保标题栏(含 X 按钮)可见
                    x = max(available.x(), min(x, available.right() - width + 1))
                    # 约束 y 确保标题栏可见
                    y = max(available.y(), min(y, available.y() + 30))

            self.setGeometry(x, y, width, height)

            # 恢复窗口状态
            window_state = state.get("window_state", "normal")
            if window_state == "maximized":
                self.showMaximized()
            elif window_state == "fullscreen":
                self.showFullScreen()
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

    def _refresh_workspace_combo(self) -> None:
        """刷新工作区间 ComboBox 列表,保持当前选中项(若仍存在)。"""
        from yolo26_app.core.workspace_manager import WorkspaceManager
        # 记录当前选中的工作区间名称
        current_name = ""
        idx = self.workspace_combo.currentIndex()
        if idx >= 0:
            current_name = self.workspace_combo.itemText(idx)
        # blockSignals 避免填充时触发 _on_workspace_changed
        self.workspace_combo.blockSignals(True)
        try:
            self.workspace_combo.clear()
            workspaces = WorkspaceManager.list_workspaces()
            for name in workspaces:
                self.workspace_combo.addItem(name)
            # 末尾追加"默认工作区间"项(数据保存到 my_project/default/)
            self.workspace_combo.addItem("默认工作区间")
            # 恢复选中(若仍存在)
            if current_name:
                i = self.workspace_combo.findText(current_name)
                if i >= 0:
                    self.workspace_combo.setCurrentIndex(i)
                else:
                    # 当前工作区间不存在了
                    self.workspace_combo.setCurrentIndex(-1)
            else:
                self.workspace_combo.setCurrentIndex(-1)
        finally:
            self.workspace_combo.blockSignals(False)

    def _rollback_workspace_combo(self, rollback_name: str) -> None:
        """回退工作区间 ComboBox 到指定名称(用 blockSignals 避免递归)。"""
        self.workspace_combo.blockSignals(True)
        if rollback_name:
            i = self.workspace_combo.findText(rollback_name)
            if i >= 0:
                self.workspace_combo.setCurrentIndex(i)
            else:
                self.workspace_combo.setCurrentIndex(-1)
        else:
            self.workspace_combo.setCurrentIndex(-1)
        self.workspace_combo.blockSignals(False)

    def _on_workspace_changed(self, index: int) -> None:
        """ComboBox 选中项变化时切换工作区间。"""
        # 记录切换前的工作区间名(用于错误回退)
        rollback_name = ""
        if self.current_project_config is not None:
            rollback_name = self.current_project_config.project_name
        if index < 0:
            return
        name = self.workspace_combo.itemText(index)
        if not name or name == "请选择工作区间":
            return
        if name == "默认工作区间":
            # flush_autosave 当前工作区间
            if self.annotate_widget is not None:
                try:
                    self.annotate_widget.flush_autosave()
                except Exception:
                    pass
            # 使用默认工作区间(my_project/default/)
            from yolo26_app.core.paths import DEFAULT_PROJECT_DIR
            from yolo26_app.core.config import ProjectConfig
            default_config = ProjectConfig(
                project_name="default",
                project_path=str(DEFAULT_PROJECT_DIR),
            )
            DEFAULT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)
            self._set_project_config(default_config)
            self._last_successful_workspace = "默认工作区间"
            self.statusbar.showMessage("默认工作区间:数据保存到 my_project/default/", 0)
            self._schedule_recovery_save()
            return
        from yolo26_app.core.workspace_manager import WorkspaceManager
        workspace_path = WorkspaceManager.get_workspace_path(name)
        # 先保存当前工作区间的未保存数据
        if self.annotate_widget is not None:
            try:
                self.annotate_widget.flush_autosave()
            except Exception:
                pass
        # 打开新工作区间
        try:
            config = ProjectManager.open_project(str(workspace_path))
        except FileNotFoundError:
            QMessageBox.critical(self, "错误", f"工作区间配置文件不存在: {workspace_path}")
            # 回退选中(用 blockSignals 避免递归)
            self._rollback_workspace_combo(rollback_name)
            return
        except Exception as e:
            QMessageBox.critical(self, "错误", f"工作区间配置文件损坏,无法打开:\n{e}")
            # 回退选中(用 blockSignals 避免递归)
            self._rollback_workspace_combo(rollback_name)
            return
        # _set_project_config 会更新 ComboBox 选中(用 blockSignals 避免递归)
        old_config = self.current_project_config
        try:
            self._set_project_config(config)
        except Exception as e:
            # 恢复 current_project_config 为切换前的值
            self.current_project_config = old_config
            QMessageBox.critical(self, "错误", f"切换工作区间失败:\n{e}")
            # 回退 ComboBox 到切换前工作区间
            self._rollback_workspace_combo(rollback_name)
            return
        self._last_successful_workspace = config.project_name
        self.statusbar.clearMessage()
        self.statusbar.showMessage(f"已切换到工作区间: {config.project_name}", 5000)
        self._refresh_recent_projects()

    def _on_refresh_workspace(self) -> None:
        """刷新工作区间列表。"""
        # 记录刷新前的当前工作区间名称
        old_name = ""
        idx = self.workspace_combo.currentIndex()
        if idx >= 0:
            old_name = self.workspace_combo.itemText(idx)
        self._refresh_workspace_combo()
        # 检查原工作区间是否还存在
        new_idx = self.workspace_combo.currentIndex()
        if old_name and new_idx < 0:
            self.statusbar.showMessage("当前工作区间已被外部删除,请重新选择", 5000)
        else:
            self.statusbar.showMessage("工作区间列表已刷新", 3000)

    def _on_new_workspace(self) -> None:
        """新建工作区间。"""
        from yolo26_app.core.workspace_manager import WorkspaceManager
        # 生成默认名称 project1, project2, ...
        default_name = self._generate_default_workspace_name()
        name, ok = QInputDialog.getText(
            self, "新建工作区间", "工作区间名称:", text=default_name
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            config = WorkspaceManager.create_workspace(name)
        except ValueError as e:
            QMessageBox.warning(self, "无法创建", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建工作区间失败:\n{e}")
            return
        # 刷新列表并选中新建的工作区间(选中会触发 _on_workspace_changed)
        self._refresh_workspace_combo()
        i = self.workspace_combo.findText(config.project_name)
        if i >= 0:
            self.workspace_combo.setCurrentIndex(i)
        self.statusbar.showMessage(f"已创建工作区间: {config.project_name}", 5000)

    def _generate_default_workspace_name(self) -> str:
        """生成不冲突的默认工作区间名 project1, project2, ..."""
        from yolo26_app.core.workspace_manager import WorkspaceManager
        existing = set(WorkspaceManager.list_workspaces())
        i = 1
        while f"project{i}" in existing:
            i += 1
        return f"project{i}"
