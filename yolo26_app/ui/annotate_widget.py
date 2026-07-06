import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRectF, QPointF, QThread, QTimer
from PyQt6.QtGui import (
    QPixmap,
    QImage,
    QIcon,
    QColor,
    QPainter,
    QBrush,
    QPolygonF,
    QKeySequence,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QPushButton,
    QScrollArea,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QLabel,
    QProgressDialog,
    QDialog,
    QComboBox,
    QGridLayout,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QLineEdit,
    QAbstractSpinBox,
)

from yolo26_app.core.annotation_canvas import AnnotationScene, AnnotationView, AnnotationItem
from yolo26_app.core.config import ClassItem, ProjectConfig
from yolo26_app.core.label_manager import LabelManager
from yolo26_app.core.persistence import write_json_atomic
from yolo26_app.core.project_manager import ProjectManager
from yolo26_app.ui import styles


class _FlipIdxDialog(QDialog):
    """flip_idx 配置对话框，用于 pose 任务导出时配置左右翻转映射"""

    def __init__(self, kpt_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配置 flip_idx（左右翻转映射）")
        self._kpt_count = kpt_count
        self._spin_boxes: List[QSpinBox] = []

        layout = QVBoxLayout(self)

        # 说明文字
        info_label = QLabel(
            "flip_idx 用于定义关键点的左右翻转映射。\n"
            "当图像水平翻转时，关键点索引 i 会映射到 flip_idx[i]。\n"
            "例如：左手腕(索引3) → 右手腕(索引4)，则 flip_idx[3]=4, flip_idx[4]=3。\n\n"
            "如果不配置，训练时将不使用水平翻转数据增强。"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 配置区域
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)

        header_idx = QLabel("关键点索引")
        header_flip = QLabel("翻转后索引")
        form_layout.addWidget(header_idx, 0, 0)
        form_layout.addWidget(header_flip, 0, 1)

        for i in range(kpt_count):
            idx_label = QLabel(f"关键点 {i}")
            spin = QSpinBox()
            spin.setRange(-1, kpt_count - 1)
            spin.setValue(-1)  # 默认 -1 表示不翻转
            spin.setSpecialValueText("不翻转")
            form_layout.addWidget(idx_label, i + 1, 0)
            form_layout.addWidget(spin, i + 1, 1)
            self._spin_boxes.append(spin)

        layout.addWidget(form_widget)

        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_flip_idx(self) -> Optional[List[int]]:
        """获取用户配置的 flip_idx，如果全部为 -1 则返回 None"""
        flip_idx = []
        for spin in self._spin_boxes:
            flip_idx.append(spin.value())

        # 如果全部为 -1（不翻转），则返回 None 表示不配置
        if all(v == -1 for v in flip_idx):
            return None
        return flip_idx


class _ClassMappingDialog(QDialog):
    def __init__(self, model_class_names: list[str], project_class_names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别映射")
        self._model_class_names = model_class_names
        self._project_class_names = project_class_names
        self._combos: list[QComboBox] = []

        layout = QGridLayout(self)

        header_model = QLabel("模型类别")
        header_project = QLabel("映射到项目类别")
        layout.addWidget(header_model, 0, 0)
        layout.addWidget(header_project, 0, 1)

        for i, name in enumerate(model_class_names):
            label = QLabel(name)
            combo = QComboBox()
            combo.addItem("跳过")
            for proj_name in project_class_names:
                combo.addItem(proj_name)
            if i < len(project_class_names) and project_class_names[i] == name:
                combo.setCurrentIndex(i + 1)
            layout.addWidget(label, i + 1, 0)
            layout.addWidget(combo, i + 1, 1)
            self._combos.append(combo)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, len(model_class_names) + 1, 0, 1, 2)

    def get_mapping(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for i, combo in enumerate(self._combos):
            idx = combo.currentIndex()
            if idx > 0:
                mapping[i] = idx - 1
        return mapping


class _SamWorker(QThread):
    encoding_done = pyqtSignal()
    prediction_done = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, predictor, task="encode", image=None, points=None, labels=None):
        super().__init__()
        self._predictor = predictor
        self._task = task
        self._image = image
        self._points = points
        self._labels = labels

    def run(self):
        use_autocast = False
        try:
            import torch
            use_autocast = torch.cuda.is_available()
        except ImportError:
            pass

        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()

        try:
            if self._task == "encode":
                with ctx:
                    self._predictor.set_image(self._image)
                self.encoding_done.emit()
            elif self._task == "predict":
                with ctx:
                    masks, scores, logits = self._predictor.predict(
                        point_coords=self._points,
                        point_labels=self._labels,
                        multimask_output=True,
                    )
                self.prediction_done.emit((masks, scores, logits))
        except Exception as e:
            self.error_occurred.emit(str(e))


class _ModelDownloadWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url: str, save_path: str, parent=None):
        super().__init__(parent)
        self._url = url
        self._save_path = save_path
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        try:
            import urllib.request
            import tempfile
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            tmp_path = self._save_path + ".tmp"
            urllib.request.urlretrieve(
                self._url, tmp_path,
                reporthook=self._download_hook
            )
            if not self._stop_flag:
                os.rename(tmp_path, self._save_path)
                self.finished.emit(self._save_path)
            else:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            self.error.emit(str(e))

    def _download_hook(self, block_num: int, block_size: int, total_size: int) -> None:
        if self._stop_flag:
            raise Exception("Download cancelled")
        if total_size > 0:
            progress = int(block_num * block_size / total_size * 100)
            self.progress.emit(min(progress, 100))


class _BatchDetectWorker(QThread):
    progress_signal = pyqtSignal(int, int)
    done_signal = pyqtSignal(dict, int)
    error_signal = pyqtSignal(str)

    def __init__(self, image_list, yolo_annotator, conf):
        super().__init__()
        self._image_list = image_list
        self._yolo_annotator = yolo_annotator
        self._conf = conf
        self._stop_flag = False

    def run(self):
        results_dict: Dict[str, List[AnnotationItem]] = {}
        total = len(self._image_list)
        try:
            for i, img_path in enumerate(self._image_list):
                if self._stop_flag:
                    break
                annotations = self._yolo_annotator.annotate(img_path, conf=self._conf)
                if annotations:
                    results_dict[img_path] = annotations
                self.progress_signal.emit(i + 1, total)
            self.done_signal.emit(results_dict, total)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._stop_flag = True


class _ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(int, QPixmap)

    def __init__(self, items: list, parent=None):
        super().__init__(parent)
        self._items = items
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True
        self.quit()
        self.wait(3000)

    def run(self) -> None:
        for row, path in self._items:
            if self._stop_flag:
                break
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.thumbnail_ready.emit(row, scaled)


class AnnotateWidget(QWidget):
    export_requested = pyqtSignal()
    state_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label_manager = LabelManager()
        self._annotations_dict: Dict[str, List[AnnotationItem]] = {}
        self._current_image_path: str = ""
        self._image_list: List[str] = []
        self._yolo_annotator = None
        self._sam_annotator = None
        self._dino_annotator = None
        self._sam_instructions_shown = False
        self._sam_encoding = False
        self._sam_worker = None
        self._batch_worker = None
        self._batch_progress = None
        self._thumb_worker: Optional[_ThumbnailWorker] = None
        self._download_worker = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(0)
        self._autosave_timer.timeout.connect(self._save_annotations_to_project)

        self._setup_ui()
        self._connect_signals()

    def _get_yolo_annotator(self):
        if self._yolo_annotator is None:
            from yolo26_app.core.auto_annotator import YOLOPreAnnotator
            self._yolo_annotator = YOLOPreAnnotator()
        return self._yolo_annotator

    def _get_sam_annotator(self):
        if self._sam_annotator is None:
            from yolo26_app.core.auto_annotator import SAMAnnotator
            self._sam_annotator = SAMAnnotator()
        return self._sam_annotator

    def _get_dino_annotator(self):
        if self._dino_annotator is None:
            from yolo26_app.core.auto_annotator import GroundingDINOAnnotator
            self._dino_annotator = GroundingDINOAnnotator()
        return self._dino_annotator

    def _sam_set_image_async(self, image_path: str) -> None:
        if self._sam_annotator is None or self._sam_annotator._predictor is None:
            return
        if self._sam_worker is not None and self._sam_worker.isRunning():
            self._sam_worker.quit()
            self._sam_worker.wait()
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            return
        self._sam_encoding = True
        self._scene._sam_encoding = True
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage("SAM 2 正在编码图像...")
        self._sam_worker = _SamWorker(self._sam_annotator._predictor, task="encode", image=image)
        self._sam_worker.encoding_done.connect(self._on_sam_encode_done)
        self._sam_worker.error_occurred.connect(self._on_sam_error)
        self._sam_worker.start()
        self._sam_worker.finished.connect(self._sam_worker.deleteLater)
        self._sam_worker.finished.connect(lambda: setattr(self, '_sam_worker', None))

    def _sam_predict_async(self) -> None:
        if self._sam_annotator is None or self._sam_annotator._predictor is None:
            return
        if self._sam_encoding:
            return
        points, labels = self._scene.get_sam_input_points()
        if points is None or len(points) == 0:
            return
        import numpy as np
        points_np = np.array(points)
        labels_np = np.array(labels)
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage("SAM 2 正在预测...")
        if self._sam_worker is not None and self._sam_worker.isRunning():
            self._sam_worker.quit()
            self._sam_worker.wait()
        self._sam_worker = _SamWorker(
            self._sam_annotator._predictor,
            task="predict",
            points=points_np,
            labels=labels_np,
        )
        self._sam_worker.prediction_done.connect(self._on_sam_predict_done)
        self._sam_worker.error_occurred.connect(self._on_sam_error)
        self._sam_worker.start()
        self._sam_worker.finished.connect(self._sam_worker.deleteLater)
        self._sam_worker.finished.connect(lambda: setattr(self, '_sam_worker', None))

    def _on_sam_encode_done(self) -> None:
        self._sam_encoding = False
        self._scene._sam_encoding = False
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage("SAM 2 编码完成，可以开始标注")

    def _on_sam_predict_done(self, result) -> None:
        masks, scores, logits = result
        self._scene.apply_sam_result(masks, scores)
        self._save_current_annotations()
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage("就绪")

    def _on_sam_error(self, error_msg: str) -> None:
        self._sam_encoding = False
        self._scene._sam_encoding = False
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage(f"SAM 2 错误: {error_msg}")

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self._toolbar = QScrollArea()
        self._toolbar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._toolbar.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._toolbar.setWidgetResizable(False)
        self._toolbar.setMinimumHeight(50)
        self._toolbar.setMaximumHeight(80)

        toolbar_container = QWidget()
        toolbar_layout = QHBoxLayout(toolbar_container)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        toolbar_layout.setSpacing(8)

        self._btn_rect = QPushButton("矩形标注")
        self._btn_rect.setCheckable(True)
        self._btn_rect.setChecked(True)
        self._btn_polygon = QPushButton("多边形标注")
        self._btn_polygon.setCheckable(True)
        self._btn_keypoint = QPushButton("关键点")
        self._btn_keypoint.setCheckable(True)
        self._btn_keypoint.setToolTip("关键点标注工具")
        self._btn_select = QPushButton("选择")
        self._btn_select.setCheckable(True)
        self._btn_delete = QPushButton("删除")
        self._btn_sam = QPushButton("SAM分割")
        self._btn_dino = QPushButton("文本检测")
        self._btn_batch = QPushButton("逐帧检测")
        self._btn_import_img = QPushButton("导入图片")
        self._btn_import_video = QPushButton("导入视频")
        self._btn_import_dir = QPushButton("导入目录")
        self._btn_export = QPushButton("导出数据集")

        toolbar_layout.addWidget(self._btn_rect)
        toolbar_layout.addWidget(self._btn_polygon)
        toolbar_layout.addWidget(self._btn_keypoint)
        toolbar_layout.addWidget(self._btn_select)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._btn_delete)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._btn_sam)
        toolbar_layout.addWidget(self._btn_dino)
        toolbar_layout.addWidget(self._btn_batch)
        toolbar_layout.addWidget(self._btn_import_img)
        toolbar_layout.addWidget(self._btn_import_video)
        toolbar_layout.addWidget(self._btn_import_dir)
        toolbar_layout.addSpacing(8)
        toolbar_layout.addWidget(self._btn_export)

        self._toolbar.setWidget(toolbar_container)

        main_layout.addWidget(self._toolbar)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        left_splitter = QSplitter(Qt.Orientation.Vertical)

        self._image_list_widget = QListWidget()
        self._image_list_widget.setIconSize(QSize(64, 64))
        left_splitter.addWidget(self._image_list_widget)

        class_group = QGroupBox("类别")
        class_layout = QVBoxLayout(class_group)

        self._class_list_widget = QListWidget()
        class_layout.addWidget(self._class_list_widget)

        btn_row = QHBoxLayout()
        self._btn_add_class = QPushButton("+")
        self._btn_add_class.setMinimumWidth(40)
        self._btn_remove_class = QPushButton("-")
        self._btn_remove_class.setMinimumWidth(40)
        btn_row.addWidget(self._btn_add_class)
        btn_row.addWidget(self._btn_remove_class)
        btn_row.addStretch()
        class_layout.addLayout(btn_row)

        left_splitter.addWidget(class_group)
        left_splitter.setSizes([400, 200])

        self._scene = AnnotationScene()
        self._view = AnnotationView(self._scene)
        self._view.setStyleSheet(styles.SCENE_BACKGROUND_STYLE)

        self._splitter.addWidget(left_splitter)
        self._splitter.addWidget(self._view)
        self._splitter.setSizes([250, 750])

        main_layout.addWidget(self._splitter)
        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        self._btn_rect.clicked.connect(lambda: self._set_tool("rect"))
        self._btn_polygon.clicked.connect(lambda: self._set_tool("polygon"))
        self._btn_keypoint.clicked.connect(lambda: self._set_tool("keypoint"))
        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_sam.clicked.connect(self._sam_annotate)
        self._btn_dino.clicked.connect(self._text_detect)
        self._btn_batch.clicked.connect(self._batch_detect)
        self._btn_import_img.clicked.connect(self._import_images)
        self._btn_import_video.clicked.connect(self._import_video)
        self._btn_import_dir.clicked.connect(self._import_directory)
        self._btn_export.clicked.connect(self._export_dataset)
        self._image_list_widget.currentItemChanged.connect(self._on_image_selected)
        self._btn_add_class.clicked.connect(self._add_class)
        self._btn_remove_class.clicked.connect(self._remove_class)
        self._class_list_widget.currentRowChanged.connect(self._on_class_selected)
        self._scene.annotations_changed.connect(self._on_annotations_changed)

        self._next_image_shortcut = QShortcut(QKeySequence("Space"), self)
        self._next_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_image_shortcut.setAutoRepeat(False)
        self._next_image_shortcut.activated.connect(self._go_to_next_image)

    def _on_annotations_changed(self) -> None:
        self._save_current_annotations()
        self._schedule_autosave()

    def _schedule_autosave(self) -> None:
        self.state_changed.emit()
        self._autosave_timer.start()

    def flush_autosave(self) -> None:
        self._autosave_timer.stop()
        self._save_current_annotations()
        self._save_annotations_to_project()

    def _go_to_next_image(self) -> None:
        if QApplication.activeModalWidget() is not None:
            return
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QAbstractSpinBox, QComboBox)):
            return

        row = self._image_list_widget.currentRow()
        if row < 0 and self._image_list_widget.count() > 0:
            self._image_list_widget.setCurrentRow(0)
        elif row < self._image_list_widget.count() - 1:
            self._image_list_widget.setCurrentRow(row + 1)

    def _set_tool(self, tool: str) -> None:
        self._btn_rect.setChecked(tool == "rect")
        self._btn_polygon.setChecked(tool == "polygon")
        self._btn_keypoint.setChecked(tool == "keypoint")
        self._btn_select.setChecked(tool == "select")
        self._scene.set_tool(tool)
        if tool == "keypoint":
            self._scene.set_kpt_count(self._get_current_kpt_count())

    def _get_current_kpt_count(self) -> int:
        idx = self._class_list_widget.currentRow()
        classes = self._label_manager.get_all_classes()
        if 0 <= idx < len(classes):
            return classes[idx].kpt_count
        return 0

    def _get_max_kpt_count(self) -> int:
        """获取所有类别中最大的关键点数量"""
        classes = self._label_manager.get_all_classes()
        max_kpt = 0
        for cls in classes:
            if cls.kpt_count > max_kpt:
                max_kpt = cls.kpt_count
        # 如果类别中没有配置，从标注数据中获取
        if max_kpt == 0:
            for anns in self._annotations_dict.values():
                for ann in anns:
                    if ann.keypoints and len(ann.keypoints) > max_kpt:
                        max_kpt = len(ann.keypoints)
        return max_kpt

    def _delete_selected(self) -> None:
        self._scene.delete_selected()
        self._save_current_annotations()

    def _copy_to_project_images(self, src_path: str) -> str:
        """将素材复制到项目 images/ 目录，返回相对项目路径的路径。

        无项目时返回原路径；已在 images/ 内返回相对路径；否则复制（重名加序号）。
        """
        window = self.window()
        if not hasattr(window, "current_project_config") or window.current_project_config is None:
            return src_path
        config = window.current_project_config
        project_path = config.project_path
        images_dir = ProjectManager.get_images_dir(config)
        images_dir.mkdir(parents=True, exist_ok=True)

        src = Path(src_path)
        # 若已在 images/ 内，直接返回相对路径
        try:
            rel = os.path.relpath(src_path, project_path)
        except ValueError:
            rel = src_path
        if os.path.isfile(os.path.join(project_path, rel)) and src_path == os.path.join(project_path, rel):
            return rel.replace(os.sep, "/")

        # 复制到 images/，重名加序号
        dest = images_dir / src.name
        if dest.resolve() == src.resolve():
            return os.path.relpath(str(dest), project_path).replace(os.sep, "/")
        counter = 1
        stem, suffix = src.stem, src.suffix
        while dest.exists():
            dest = images_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        import shutil
        shutil.copy2(str(src), str(dest))
        return os.path.relpath(str(dest), project_path).replace(os.sep, "/")

    def _import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "导入图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not files:
            return
        for f in files:
            stored = self._copy_to_project_images(f)
            if stored not in self._image_list:
                self._image_list.append(stored)
                self._annotations_dict.setdefault(stored, [])
                self._add_image_item(stored)
        self._start_thumbnail_loading()
        self._schedule_autosave()

    def _import_directory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not dir_path:
            return
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files: List[str] = []
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    image_files.append(os.path.join(root, f))
        image_files.sort()
        if not image_files:
            QMessageBox.warning(self, "提示", "所选目录中没有找到图片文件")
            return
        for path in image_files:
            stored = self._copy_to_project_images(path)
            if stored not in self._image_list:
                self._image_list.append(stored)
                self._annotations_dict.setdefault(stored, [])
                self._add_image_item(stored)
        self._start_thumbnail_loading()
        self._image_list_widget.setCurrentRow(0)
        self._schedule_autosave()

    def _import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入视频", "", "Videos (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        import cv2
        window = self.window()
        if hasattr(window, "current_project_config") and window.current_project_config is not None:
            temp_dir = str(ProjectManager.get_images_dir(window.current_project_config))
            os.makedirs(temp_dir, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp(prefix="yolo26_frames_")
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        interval = max(1, int(fps))
        frame_idx = 0
        saved_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_path = os.path.join(temp_dir, f"frame_{saved_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                stored = self._copy_to_project_images(out_path) if (hasattr(window, "current_project_config") and window.current_project_config is not None) else out_path
                self._image_list.append(stored)
                self._annotations_dict.setdefault(stored, [])
                self._add_image_item(stored)
                saved_idx += 1
            frame_idx += 1
        cap.release()
        self._start_thumbnail_loading()
        self._schedule_autosave()

    def _add_image_item(self, image_path: str) -> None:
        item = QListWidgetItem(os.path.basename(image_path))
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        item.setToolTip(image_path)
        self._image_list_widget.addItem(item)

    def _start_thumbnail_loading(self) -> None:
        items = []
        for row in range(self._image_list_widget.count()):
            item = self._image_list_widget.item(row)
            path = item.data(Qt.ItemDataRole.UserRole)
            items.append((row, path))
        if items:
            if hasattr(self, '_thumb_worker') and self._thumb_worker is not None:
                try:
                    self._thumb_worker.stop()
                except RuntimeError:
                    pass
            self._thumb_worker = _ThumbnailWorker(items, self)
            self._thumb_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
            self._thumb_worker.finished.connect(self._thumb_worker.deleteLater)
            self._thumb_worker.finished.connect(lambda: setattr(self, '_thumb_worker', None))
            self._thumb_worker.start()

    def _on_thumbnail_ready(self, row: int, pixmap: QPixmap) -> None:
        if 0 <= row < self._image_list_widget.count():
            item = self._image_list_widget.item(row)
            item.setIcon(QIcon(pixmap))

    def _on_image_selected(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        self._save_current_annotations()
        self._schedule_autosave()
        image_path = current.data(Qt.ItemDataRole.UserRole)
        self._current_image_path = image_path
        self._load_image(image_path)
        if (self._scene.current_tool == "sam"
                and self._sam_annotator is not None
                and self._sam_annotator._predictor is not None):
            self._sam_set_image_async(image_path)

    def _load_image(self, image_path: str) -> None:
        self._scene.clear_annotations()
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        from PyQt6.QtWidgets import QGraphicsPixmapItem
        from PyQt6.QtGui import QPixmap as QPxm
        img_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))

        anns = self._annotations_dict.get(image_path, [])
        self._scene.load_annotations(anns)
        self._view.fit_to_item()

    def _save_current_annotations(self) -> None:
        if self._current_image_path:
            self._annotations_dict[self._current_image_path] = self._scene.get_annotations()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Up:
            row = self._image_list_widget.currentRow()
            if row > 0:
                self._image_list_widget.setCurrentRow(row - 1)
            return
        elif event.key() == Qt.Key.Key_Down:
            row = self._image_list_widget.currentRow()
            if row < self._image_list_widget.count() - 1:
                self._image_list_widget.setCurrentRow(row + 1)
            return

        if self._scene.current_tool == "sam":
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                self._sam_predict_async()
                return
            elif event.key() == Qt.Key.Key_Escape:
                self._scene.clear_sam_points()
                self._scene.set_tool("select")
                return
        super().keyPressEvent(event)

    def _add_class(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("添加类别")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit()
        kpt_spin = QSpinBox()
        kpt_spin.setRange(0, 100)
        kpt_spin.setValue(0)
        layout.addRow("类别名称:", name_edit)
        layout.addRow("关键点数量:", kpt_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_edit.text().strip()
            kpt_count = kpt_spin.value()
            if not name:
                return
            if self._label_manager.get_class_index(name) >= 0:
                QMessageBox.warning(self, "重复", f"类别 '{name}' 已存在")
                return
            self._label_manager.add_class(name, kpt_count=kpt_count)
            self._update_class_list()
            self._update_scene_colors()
            self._persist_classes()

    def _remove_class(self) -> None:
        row = self._class_list_widget.currentRow()
        if row < 0:
            return
        self._label_manager.remove_class(row)
        self._update_class_list()
        self._update_scene_colors()
        self._persist_classes()

    def _on_class_selected(self, row: int) -> None:
        if row >= 0:
            self._scene.set_current_class(row)

    def _update_class_list(self) -> None:
        self._class_list_widget.clear()
        classes = self._label_manager.get_all_classes()
        for cls_item in classes:
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(cls_item.color))
            label = cls_item.name + (f" ({cls_item.kpt_count}pt)" if cls_item.kpt_count > 0 else "")
            item = QListWidgetItem(QIcon(pixmap), label)
            self._class_list_widget.addItem(item)

    def _update_scene_colors(self) -> None:
        colors = [c.color for c in self._label_manager.get_all_classes()]
        self._scene.set_class_colors(colors)
        self._scene.set_class_names([c.name for c in self._label_manager.get_all_classes()])

    def _persist_classes(self) -> None:
        """将类别列表保存到项目配置文件"""
        self.state_changed.emit()
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return
        project_config = window.current_project_config
        if project_config is None:
            return
        project_config.classes = self._label_manager.get_all_classes()
        try:
            from pathlib import Path
            config_path = Path(project_config.project_path) / "project_config.json"
            project_config.save(config_path)
        except Exception:
            pass

    def _sam_annotate(self) -> None:
        if not self._current_image_path:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        sam = self._get_sam_annotator()
        if not sam.available:
            QMessageBox.information(
                self, "提示",
                "请先安装 SAM 2:\npip install sam2\n\n"
                "并下载模型权重:\nhttps://github.com/facebookresearch/segment-anything-2#download-checkpoints"
            )
            return
        if sam._predictor is None:
            model_info = None
            # SAM2 模型统一存到 system_model/sam2/
            from yolo26_app.core.paths import SYSTEM_MODEL_SUBDIRS
            sam2_dir = str(SYSTEM_MODEL_SUBDIRS["sam2"])
            os.makedirs(sam2_dir, exist_ok=True)
            scan_dirs = [sam2_dir]
            if self._current_image_path:
                scan_dirs.append(os.path.dirname(self._current_image_path))
            for d in scan_dirs:
                model_info = sam.scan_model_file(d)
                if model_info:
                    break
            if model_info:
                model_path, model_type, config_path = model_info
            else:
                start_dir = scan_dirs[0] if scan_dirs else ""
                from yolo26_app.core.auto_annotator import SAM2_MODEL_URLS
                model_names = list(SAM2_MODEL_URLS.keys())
                choice, ok = QInputDialog.getItem(
                    self, "SAM 2 模型",
                    "未找到 SAM 2 模型文件。\n请选择要下载的模型或点击取消手动选择：",
                    model_names, 0, False
                )
                if ok and choice:
                    url = SAM2_MODEL_URLS[choice]
                    save_dir = scan_dirs[0] if scan_dirs else os.getcwd()
                    save_path = os.path.join(save_dir, f"{choice}.pt")
                    progress_dlg = QProgressDialog(f"正在下载 {choice}...", "取消", 0, 100, self)
                    progress_dlg.setWindowModality(Qt.WindowModality.WindowModal)
                    self._download_worker = _ModelDownloadWorker(url, save_path, self)
                    self._download_worker.progress.connect(progress_dlg.setValue)
                    self._download_worker.finished.connect(lambda p: (progress_dlg.close(), self._on_sam_model_downloaded(p, sam, device)))
                    self._download_worker.error.connect(lambda e: (progress_dlg.close(), QMessageBox.warning(self, "下载失败", str(e))))
                    progress_dlg.canceled.connect(self._download_worker.stop)
                    self._download_worker.start()
                    self._download_worker.finished.connect(self._download_worker.deleteLater)
                    self._download_worker.finished.connect(lambda: setattr(self, '_download_worker', None))
                    progress_dlg.exec()
                    return
                model_path, _ = QFileDialog.getOpenFileName(
                    self, "选择 SAM 2 模型权重", start_dir, "PyTorch Weights (*.pt *.pth)"
                )
                if not model_path:
                    return
                config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
                model_type = "sam2.1_hiera_s"
                filename = os.path.basename(model_path).lower()
                if "hiera_l" in filename or "hiera-large" in filename:
                    config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
                    model_type = "sam2.1_hiera_l"
                elif "hiera_b" in filename or "hiera-base" in filename:
                    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                    model_type = "sam2.1_hiera_b+"
                elif "hiera_t" in filename or "hiera-tiny" in filename:
                    config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"
                    model_type = "sam2.1_hiera_t"
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            window = self.window()
            if hasattr(window, "statusbar"):
                window.statusbar.showMessage("SAM 2 正在加载模型...")
            QApplication.processEvents()
            if not sam.load_model(model_path, config_path, device):
                QMessageBox.critical(self, "错误", "SAM 模型加载失败")
                if hasattr(window, "statusbar"):
                    window.statusbar.showMessage("就绪")
                return
            if hasattr(window, "statusbar"):
                window.statusbar.showMessage("SAM 2 模型加载完成")
        self._scene.set_sam_annotator(sam)
        self._scene.set_tool("sam")
        self._sam_set_image_async(self._current_image_path)
        if not self._sam_instructions_shown:
            self._sam_instructions_shown = True
            QMessageBox.information(
                self, "SAM 2 分割",
                "已进入 SAM 2 分割模式\n\n"
                "左键点击 = 前景点（绿色）\n"
                "右键点击 = 背景点（红色）\n"
                "按 Enter 键确认分割\n"
                "按 Esc 键取消"
            )

    def _on_sam_model_downloaded(self, model_path: str, sam: 'SAMAnnotator', device: str) -> None:
        model_info = sam.scan_model_file(os.path.dirname(model_path))
        if model_info:
            _, _, config_path = model_info
        else:
            config_path = "configs/sam2.1/sam2.1_hiera_s.yaml"
        window = self.window()
        window.statusbar.showMessage("SAM 2 正在加载模型...")
        if sam.load_model(model_path, config_path, device):
            window.statusbar.showMessage("SAM 2 模型加载完成")
            self._scene.set_sam_annotator(sam)
            self._scene.current_tool = "sam"
            self._sam_set_image_async(self._current_image_path)
            QMessageBox.information(
                self, "SAM 2 分割",
                "已进入 SAM 2 分割模式\n\n"
                "左键点击：添加前景点\n"
                "右键点击：添加背景点\n"
                "双击：确认分割结果\n"
                "Esc：取消当前分割\n"
                "↑↓键：切换图片"
            )
        else:
            QMessageBox.warning(self, "错误", "SAM 2 模型加载失败")

    def _text_detect(self) -> None:
        if not self._current_image_path:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        dino = self._get_dino_annotator()
        if not dino.available:
            QMessageBox.information(
                self, "提示",
                "请先安装 Grounding DINO:\npip install groundingdino\n\n"
                "详见: https://github.com/IDEA-Research/GroundingDINO"
            )
            return
        if dino._model is None:
            config_path, _ = QFileDialog.getOpenFileName(
                self, "选择 Grounding DINO 配置文件", "", "Config (*.py *.yaml *.yml)"
            )
            if not config_path:
                return
            weights_path, _ = QFileDialog.getOpenFileName(
                self, "选择 Grounding DINO 权重文件", "", "PyTorch Weights (*.pt *.pth)"
            )
            if not weights_path:
                return
            if not dino.load_model(config_path, weights_path):
                QMessageBox.critical(self, "错误", "Grounding DINO 模型加载失败")
                return
        text, ok = QInputDialog.getText(
            self, "文本检测", "输入目标描述 (用 . 分隔，如 car . person .):"
        )
        if not ok or not text.strip():
            return
        annotations = dino.detect(self._current_image_path, text.strip())
        if not annotations:
            QMessageBox.information(self, "提示", "未检测到目标")
            return
        for ann in annotations:
            self._scene._annotations.append(ann)
            self._scene._draw_annotation(ann, len(self._scene._annotations) - 1)
        self._scene.annotations_changed.emit()
        self._save_current_annotations()

    def set_yolo_model(self, model) -> None:
        self._get_yolo_annotator().set_model(model)

    def _batch_detect(self) -> None:
        yolo = self._get_yolo_annotator()
        if yolo._model is None:
            QMessageBox.information(self, "提示", "请先在测试页面加载模型")
            return
        if not self._image_list:
            QMessageBox.warning(self, "提示", "请先导入图片")
            return
        conf, ok = QInputDialog.getDouble(self, "逐帧检测", "置信度阈值:", 0.25, 0.01, 1.0, 2)
        if not ok:
            return
        total = len(self._image_list)
        progress = QProgressDialog("正在处理图片...", "取消", 0, total, self)
        progress.setWindowTitle("逐帧检测")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        self._batch_progress = progress
        self._batch_worker = _BatchDetectWorker(self._image_list, yolo, conf)
        self._batch_worker.progress_signal.connect(progress.setValue)
        self._batch_worker.done_signal.connect(self._on_batch_done)
        self._batch_worker.error_signal.connect(
            lambda msg: QMessageBox.critical(self, "错误", msg)
        )
        progress.canceled.connect(self._batch_worker.stop)
        self._batch_worker.start()
        self._batch_worker.finished.connect(self._batch_worker.deleteLater)
        self._batch_worker.finished.connect(lambda: setattr(self, '_batch_worker', None))

    def _on_batch_done(self, results_dict: dict, total: int) -> None:
        if self._batch_progress is not None:
            self._batch_progress.close()
            self._batch_progress = None
        for img_path, anns in results_dict.items():
            self._annotations_dict[img_path] = anns
        if self._current_image_path in self._annotations_dict:
            self._scene.clear_annotations()
            for ann in self._annotations_dict[self._current_image_path]:
                self._scene._annotations.append(ann)
                self._scene._draw_annotation(ann, len(self._scene._annotations) - 1)
            self._scene.annotations_changed.emit()
        else:
            self._schedule_autosave()
        detected_count = len(results_dict)
        QMessageBox.information(
            self, "逐帧检测完成",
            f"共处理 {total} 张图片\n检测到目标: {detected_count} 张"
        )

    def _export_dataset(self) -> None:
        self._save_current_annotations()
        classes = self._label_manager.get_all_classes()
        if not classes:
            QMessageBox.warning(self, "导出", "请先添加类别")
            return
        has_annotations = any(len(v) > 0 for v in self._annotations_dict.values())
        if not has_annotations:
            QMessageBox.warning(self, "导出", "没有标注数据可导出")
            return
        window = self.window()
        if getattr(window, "current_project_config", None) is not None:
            output_dir = str(ProjectManager.get_dataset_dir(window.current_project_config))
        else:
            output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if not output_dir:
                return

        flip_idx: Optional[List[int]] = None  # flip_idx 配置，默认不配置

        has_polygon = any(
            a.item_type == "polygon"
            for anns in self._annotations_dict.values()
            for a in anns
        )
        task = "detect"
        if has_polygon:
            items = ["detect — 多边形自动转为矩形框", "segment — 保留多边形用于分割训练", "pose — 关键点姿态格式", "classify — 分类任务（使用目录结构）"]
            item, ok = QInputDialog.getItem(
                self, "选择导出格式",
                "检测到多边形标注，请选择导出格式：",
                items, 0, False,
            )
            if not ok:
                return
            if "segment" in item:
                task = "segment"
            elif "pose" in item:
                task = "pose"
                # pose 任务时询问用户配置 flip_idx
                kpt_count = self._get_max_kpt_count()
                if kpt_count > 0:
                    flip_dlg = _FlipIdxDialog(kpt_count, self)
                    result = flip_dlg.exec()
                    if result == QDialog.DialogCode.Accepted:
                        flip_idx = flip_dlg.get_flip_idx()
                    else:
                        flip_idx = None  # 用户取消，不配置 flip_idx
                else:
                    flip_idx = None
            elif "classify" in item:
                task = "classify"
                QMessageBox.information(
                    self, "classify 导出说明",
                    "classify 任务使用目录结构导出：\n"
                    "- train/<class_name>/<image_files>\n"
                    "- val/<class_name>/<image_files>\n\n"
                    "图片将根据其主要类别（标注数量最多的类别）复制到对应目录。\n"
                    "不生成 labels 目录和 .txt 标签文件。"
                )
            else:
                task = "detect"
        else:
            window = self.window()
            if getattr(window, "train_widget", None) is not None:
                task = window.train_widget.task_combo.currentText()
            # 如果当前任务是 classify，显示提示
            if task == "classify":
                QMessageBox.information(
                    self, "classify 导出说明",
                    "classify 任务使用目录结构导出：\n"
                    "- train/<class_name>/<image_files>\n"
                    "- val/<class_name>/<image_files>\n\n"
                    "图片将根据其主要类别（标注数量最多的类别）复制到对应目录。\n"
                    "不生成 labels 目录和 .txt 标签文件。"
                )
            # pose 任务时询问用户配置 flip_idx
            elif task == "pose":
                kpt_count = self._get_max_kpt_count()
                if kpt_count > 0:
                    flip_dlg = _FlipIdxDialog(kpt_count, self)
                    result = flip_dlg.exec()
                    if result == QDialog.DialogCode.Accepted:
                        flip_idx = flip_dlg.get_flip_idx()
                    else:
                        flip_idx = None  # 用户取消，不配置 flip_idx

        try:
            from yolo26_app.core.yolo_exporter import YOLOExporter

            yaml_path, stats = YOLOExporter.export_dataset(
                self._annotations_dict, classes, output_dir, task=task, flip_idx=flip_idx
            )
            msg = f"导出完成!\n训练集: {stats['train_count']} 张\n验证集: {stats['val_count']} 张\n跳过: {stats['skipped_count']} 张"
            QMessageBox.information(self, "导出成功", msg)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def load_project_classes(self, classes: List[ClassItem]) -> None:
        for c in classes:
            if self._label_manager.get_class_index(c.name) < 0:
                self._label_manager.add_class(c.name)
        self._update_class_list()
        self._update_scene_colors()

    def get_label_manager(self) -> LabelManager:
        return self._label_manager

    def set_project_config(self, config: ProjectConfig) -> None:
        self._autosave_timer.stop()
        self._current_image_path = ""
        self._image_list.clear()
        self._annotations_dict.clear()
        self._image_list_widget.clear()
        self._scene.clear_annotations()
        self._label_manager.load_from_project(config)
        self._update_class_list()
        self._update_scene_colors()
        self._load_annotations_from_project()

    def save_state(self) -> dict:
        self._save_current_annotations()
        serialized: Dict[str, list] = {}
        for path, anns in self._annotations_dict.items():
            items = []
            for ann in anns:
                d = {"class_index": ann.class_index, "item_type": ann.item_type}
                if ann.item_type == "rect" and ann.rect is not None:
                    d["rect"] = [ann.rect.x(), ann.rect.y(), ann.rect.width(), ann.rect.height()]
                elif ann.item_type == "polygon" and ann.polygon is not None:
                    d["polygon"] = [[ann.polygon.at(i).x(), ann.polygon.at(i).y()]
                                    for i in range(ann.polygon.size())]
                if ann.keypoints:
                    d["keypoints"] = [[pt.x(), pt.y()] for pt in ann.keypoints]
                items.append(d)
            serialized[path] = items
        return {
            "annotations": serialized,
            "image_list": self._image_list,
            "current_image_path": self._current_image_path,
            "classes": [item.to_dict() for item in self._label_manager.get_all_classes()],
        }

    def restore_state(self, state: dict) -> None:
        recovered_classes = [
            ClassItem.from_dict(item)
            for item in state.get("classes", [])
            if isinstance(item, dict)
        ]
        if recovered_classes:
            self._label_manager.load_from_project(ProjectConfig(classes=recovered_classes))
            self._update_class_list()
            self._update_scene_colors()
        annotations_data = state.get("annotations", {})
        self._image_list = state.get("image_list", [])
        self._annotations_dict.clear()
        for path, items in annotations_data.items():
            anns: List[AnnotationItem] = []
            for d in items:
                rect = QRectF(*d["rect"]) if "rect" in d else QRectF()
                polygon = QPolygonF()
                if "polygon" in d:
                    for pt in d["polygon"]:
                        polygon.append(QPointF(pt[0], pt[1]))
                keypoints = [QPointF(pt[0], pt[1]) for pt in d.get("keypoints", [])]
                anns.append(AnnotationItem(
                    class_index=d.get("class_index", 0),
                    rect=rect,
                    polygon=polygon if "polygon" in d else QPolygonF(),
                    item_type=d.get("item_type", "rect"),
                    keypoints=keypoints,
                ))
            self._annotations_dict[path] = anns
        self._image_list_widget.clear()
        for img_path in self._image_list:
            self._add_image_item(img_path)
        self._start_thumbnail_loading()
        current_path = state.get("current_image_path", "")
        if current_path in self._image_list:
            self._image_list_widget.setCurrentRow(self._image_list.index(current_path))

    def _save_annotations_to_project(self) -> bool:
        if self._current_image_path:
            self._save_current_annotations()
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return False
        config = window.current_project_config
        if config is None:
            return False
        if not self._annotations_dict and not self._image_list:
            return False
        project_path = config.project_path
        serialized: Dict[str, list] = {}
        for path, anns in self._annotations_dict.items():
            items = []
            for ann in anns:
                d = {"class_index": ann.class_index, "item_type": ann.item_type}
                if ann.item_type == "rect" and ann.rect is not None:
                    d["rect"] = [ann.rect.x(), ann.rect.y(), ann.rect.width(), ann.rect.height()]
                elif ann.item_type == "polygon" and ann.polygon is not None:
                    d["polygon"] = [[ann.polygon.at(i).x(), ann.polygon.at(i).y()]
                                    for i in range(ann.polygon.size())]
                if ann.keypoints:
                    d["keypoints"] = [[pt.x(), pt.y()] for pt in ann.keypoints]
                items.append(d)
            rel_path = os.path.relpath(path, project_path) if os.path.isabs(path) else path
            serialized[rel_path] = items
        data = {
            "image_list": [os.path.relpath(p, project_path) if os.path.isabs(p) else p for p in self._image_list],
            "annotations": serialized,
            "current_image_path": os.path.relpath(self._current_image_path, project_path) if self._current_image_path and os.path.isabs(self._current_image_path) else self._current_image_path,
        }
        annotations_path = ProjectManager.get_annotations_path(config)
        try:
            write_json_atomic(annotations_path, data)
            return True
        except (PermissionError, OSError):
            if hasattr(window, "statusbar"):
                window.statusbar.showMessage("标注自动保存失败，请检查项目目录写入权限", 5000)
            return False

    def _load_annotations_from_project(self) -> None:
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return
        config = window.current_project_config
        if config is None:
            return
        project_path = config.project_path
        annotations_path = ProjectManager.get_annotations_path(config)
        if not annotations_path.exists():
            return
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        image_list = data.get("image_list", [])
        # 相对路径还原为绝对路径，绝对路径直接用（老项目兼容）
        restored_list = []
        for p in image_list:
            if os.path.isabs(p):
                restored_list.append(p)
            else:
                restored_list.append(os.path.join(project_path, p))
        image_list = restored_list
        annotations_data = data.get("annotations", {})
        valid_images = [p for p in image_list if os.path.isfile(p)]
        self._image_list = valid_images
        self._annotations_dict.clear()
        for path, items in annotations_data.items():
            if os.path.isabs(path):
                abs_path = path
            else:
                abs_path = os.path.join(project_path, path)
            if not os.path.isfile(abs_path):
                continue
            anns: List[AnnotationItem] = []
            for d in items:
                rect = QRectF(*d["rect"]) if "rect" in d else QRectF()
                polygon = QPolygonF()
                if "polygon" in d:
                    for pt in d["polygon"]:
                        polygon.append(QPointF(pt[0], pt[1]))
                keypoints = [QPointF(pt[0], pt[1]) for pt in d.get("keypoints", [])]
                anns.append(AnnotationItem(
                    class_index=d.get("class_index", 0),
                    rect=rect,
                    polygon=polygon if "polygon" in d else QPolygonF(),
                    item_type=d.get("item_type", "rect"),
                    keypoints=keypoints,
                ))
            self._annotations_dict[abs_path] = anns
        self._image_list_widget.clear()
        for img_path in self._image_list:
            self._add_image_item(img_path)
        self._start_thumbnail_loading()
        current_path = data.get("current_image_path", "")
        if current_path and not os.path.isabs(current_path):
            current_path = os.path.join(project_path, current_path)
        if current_path in self._image_list:
            self._image_list_widget.setCurrentRow(self._image_list.index(current_path))
        elif self._image_list:
            self._image_list_widget.setCurrentRow(0)
