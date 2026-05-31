import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRectF, QPointF, QThread
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor, QPainter, QBrush, QPolygonF
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
    QToolBar,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QLabel,
    QProgressDialog,
    QDialog,
    QComboBox,
    QGridLayout,
    QDialogButtonBox,
)

from yolo26_app.core.annotation_canvas import AnnotationScene, AnnotationView, AnnotationItem
from yolo26_app.core.config import ClassItem, ProjectConfig
from yolo26_app.core.label_manager import LabelManager
from yolo26_app.core.project_manager import ProjectManager
from yolo26_app.core.yolo_exporter import YOLOExporter


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
        try:
            if self._task == "encode":
                self._predictor.set_image(self._image)
                self.encoding_done.emit()
            elif self._task == "predict":
                masks, scores, logits = self._predictor.predict(
                    point_coords=self._points,
                    point_labels=self._labels,
                    multimask_output=True,
                )
                self.prediction_done.emit((masks, scores, logits))
        except Exception as e:
            self.error_occurred.emit(str(e))


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


class AnnotateWidget(QWidget):
    export_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label_manager = LabelManager()
        self._annotations_dict: Dict[str, List[AnnotationItem]] = {}
        self._current_image_path: str = ""
        self._image_list: List[str] = []
        self._yolo_annotator = None
        self._sam_annotator = None
        self._video_tracker = None
        self._dino_annotator = None
        self._sam_instructions_shown = False
        self._sam_encoding = False
        self._sam_worker = None
        self._batch_worker = None
        self._batch_progress = None

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

    def _get_video_tracker(self):
        if self._video_tracker is None:
            from yolo26_app.core.auto_annotator import VideoTracker
            self._video_tracker = VideoTracker()
            tracker_name = VideoTracker.get_tracker_name()
            self._btn_track.setText(f"视频追踪({tracker_name})")
        return self._video_tracker

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
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._toolbar = QToolBar()
        self._toolbar.setIconSize(QSize(24, 24))
        self._toolbar.setMovable(False)

        self._btn_rect = QPushButton("矩形标注")
        self._btn_rect.setCheckable(True)
        self._btn_rect.setChecked(True)
        self._btn_polygon = QPushButton("多边形标注")
        self._btn_polygon.setCheckable(True)
        self._btn_select = QPushButton("选择")
        self._btn_select.setCheckable(True)
        self._btn_delete = QPushButton("删除")
        self._btn_auto_annotate = QPushButton("YOLO预标注")
        self._btn_sam = QPushButton("SAM分割")
        self._btn_track = QPushButton("视频追踪")
        self._btn_dino = QPushButton("文本检测")
        self._btn_batch = QPushButton("逐帧检测")
        self._btn_import_img = QPushButton("导入图片")
        self._btn_import_video = QPushButton("导入视频")
        self._btn_import_dir = QPushButton("导入目录")
        self._btn_export = QPushButton("导出数据集")

        self._toolbar.addWidget(self._btn_rect)
        self._toolbar.addWidget(self._btn_polygon)
        self._toolbar.addWidget(self._btn_select)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._btn_delete)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._btn_auto_annotate)
        self._toolbar.addWidget(self._btn_sam)
        self._toolbar.addWidget(self._btn_track)
        self._toolbar.addWidget(self._btn_dino)
        self._toolbar.addWidget(self._btn_batch)
        self._toolbar.addWidget(self._btn_import_img)
        self._toolbar.addWidget(self._btn_import_video)
        self._toolbar.addWidget(self._btn_import_dir)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._btn_export)

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
        self._btn_add_class.setFixedWidth(40)
        self._btn_remove_class = QPushButton("-")
        self._btn_remove_class.setFixedWidth(40)
        btn_row.addWidget(self._btn_add_class)
        btn_row.addWidget(self._btn_remove_class)
        btn_row.addStretch()
        class_layout.addLayout(btn_row)

        left_splitter.addWidget(class_group)
        left_splitter.setSizes([400, 200])

        self._scene = AnnotationScene()
        self._view = AnnotationView(self._scene)

        self._splitter.addWidget(left_splitter)
        self._splitter.addWidget(self._view)
        self._splitter.setSizes([250, 750])

        main_layout.addWidget(self._splitter)
        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        self._btn_rect.clicked.connect(lambda: self._set_tool("rect"))
        self._btn_polygon.clicked.connect(lambda: self._set_tool("polygon"))
        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_auto_annotate.clicked.connect(self._auto_annotate)
        self._btn_sam.clicked.connect(self._sam_annotate)
        self._btn_track.clicked.connect(self._video_track)
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

    def _set_tool(self, tool: str) -> None:
        self._btn_rect.setChecked(tool == "rect")
        self._btn_polygon.setChecked(tool == "polygon")
        self._btn_select.setChecked(tool == "select")
        self._scene.set_tool(tool)

    def _delete_selected(self) -> None:
        self._scene.delete_selected()
        self._save_current_annotations()

    def _import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "导入图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not files:
            return
        for f in files:
            if f not in self._image_list:
                self._image_list.append(f)
                self._annotations_dict.setdefault(f, [])
                self._add_image_item(f)

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
            if path not in self._image_list:
                self._image_list.append(path)
                self._annotations_dict.setdefault(path, [])
                self._add_image_item(path)
        self._image_list_widget.setCurrentRow(0)
        self._on_image_selected(self._image_list_widget.currentItem(), None)

    def _import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入视频", "", "Videos (*.mp4 *.avi *.mkv *.mov)"
        )
        if not path:
            return
        import cv2
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
                self._image_list.append(out_path)
                self._annotations_dict.setdefault(out_path, [])
                self._add_image_item(out_path)
                saved_idx += 1
            frame_idx += 1
        cap.release()

    def _add_image_item(self, image_path: str) -> None:
        item = QListWidgetItem(os.path.basename(image_path))
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            item.setIcon(QIcon(scaled))
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        item.setToolTip(image_path)
        self._image_list_widget.addItem(item)

    def _on_image_selected(self, current: Optional[QListWidgetItem], previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        self._save_annotations_to_project()
        self._save_current_annotations()
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
        name, ok = QInputDialog.getText(self, "添加类别", "类别名称:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._label_manager.get_class_index(name) >= 0:
            QMessageBox.warning(self, "重复", f"类别 '{name}' 已存在")
            return
        class_item = self._label_manager.add_class(name)
        self._update_class_list()
        self._update_scene_colors()

    def _remove_class(self) -> None:
        row = self._class_list_widget.currentRow()
        if row < 0:
            return
        self._label_manager.remove_class(row)
        self._update_class_list()
        self._update_scene_colors()

    def _on_class_selected(self, row: int) -> None:
        if row >= 0:
            self._scene.set_current_class(row)

    def _update_class_list(self) -> None:
        self._class_list_widget.clear()
        classes = self._label_manager.get_all_classes()
        for cls_item in classes:
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(cls_item.color))
            item = QListWidgetItem(QIcon(pixmap), cls_item.name)
            self._class_list_widget.addItem(item)

    def _update_scene_colors(self) -> None:
        colors = [c.color for c in self._label_manager.get_all_classes()]
        self._scene.set_class_colors(colors)
        self._scene.set_class_names([c.name for c in self._label_manager.get_all_classes()])

    def _auto_annotate(self) -> None:
        if not self._current_image_path:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        yolo = self._get_yolo_annotator()
        if yolo._model is None:
            QMessageBox.information(self, "提示", "请先在测试页面加载 YOLO 模型")
            return
        conf, ok = QInputDialog.getDouble(self, "YOLO 预标注", "置信度阈值:", 0.25, 0.01, 1.0, 2)
        if not ok:
            return
        annotations = yolo.annotate(self._current_image_path, conf=conf)
        if not annotations:
            QMessageBox.information(self, "提示", "未检测到目标")
            return

        model_names_dict = yolo._model.names if yolo._model is not None else {}
        model_class_names = [model_names_dict.get(i, str(i)) for i in range(len(model_names_dict))]
        project_class_names = [c.name for c in self._label_manager.get_all_classes()]

        if model_class_names and project_class_names and model_class_names != project_class_names:
            dialog = _ClassMappingDialog(model_class_names, project_class_names, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            mapping = dialog.get_mapping()
            filtered = []
            for ann in annotations:
                if ann.class_index in mapping:
                    ann.class_index = mapping[ann.class_index]
                    filtered.append(ann)
            annotations = filtered

        for ann in annotations:
            self._scene._annotations.append(ann)
            self._scene._draw_annotation(ann, len(self._scene._annotations) - 1)
        self._scene.annotations_changed.emit()
        self._save_current_annotations()

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
            scan_dirs = [os.getcwd()]
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

    def _video_track(self) -> None:
        if len(self._image_list) < 2:
            QMessageBox.warning(self, "提示", "至少需要 2 张图片才能进行视频追踪")
            return
        current_annotations = self._scene.get_annotations()
        rect_anns = [a for a in current_annotations if a.item_type == "rect"]
        if not rect_anns:
            QMessageBox.warning(self, "提示", "当前帧没有矩形标注，请先标注目标")
            return
        max_frames, ok = QInputDialog.getInt(
            self, "视频追踪", "追踪帧数 (0=全部):", 0, 0, len(self._image_list)
        )
        if not ok:
            return
        current_idx = self._image_list.index(self._current_image_path) if self._current_image_path in self._image_list else 0
        remaining_paths = self._image_list[current_idx:]
        result = self._get_video_tracker().track_frames(remaining_paths, rect_anns, max_frames)
        if len(result) <= 1:
            QMessageBox.information(self, "提示", "追踪失败，无法传播标注")
            return
        count = 0
        for frame_offset, anns in result.items():
            if frame_offset == 0:
                continue
            img_idx = current_idx + frame_offset
            if img_idx < len(self._image_list):
                img_path = self._image_list[img_idx]
                self._annotations_dict[img_path] = anns
                count += 1
        QMessageBox.information(self, "追踪完成", f"已将标注传播到 {count} 帧")

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
        output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return

        has_polygon = any(
            a.item_type == "polygon"
            for anns in self._annotations_dict.values()
            for a in anns
        )
        task = "detect"
        if has_polygon:
            items = ["detect — 多边形自动转为矩形框", "segment — 保留多边形用于分割训练"]
            item, ok = QInputDialog.getItem(
                self, "选择导出格式",
                "检测到多边形标注，请选择导出格式：",
                items, 0, False,
            )
            if not ok:
                return
            task = "segment" if "segment" in item else "detect"
        else:
            window = self.window()
            if hasattr(window, "train_widget"):
                task = window.train_widget.task_combo.currentText()

        try:
            yaml_path, stats = YOLOExporter.export_dataset(
                self._annotations_dict, classes, output_dir, task=task
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
                items.append(d)
            serialized[path] = items
        return {"annotations": serialized, "image_list": self._image_list}

    def restore_state(self, state: dict) -> None:
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
                anns.append(AnnotationItem(
                    class_index=d.get("class_index", 0),
                    rect=rect,
                    polygon=polygon if "polygon" in d else QPolygonF(),
                    item_type=d.get("item_type", "rect"),
                ))
            self._annotations_dict[path] = anns
        self._image_list_widget.clear()
        for img_path in self._image_list:
            self._add_image_item(img_path)

    def _save_annotations_to_project(self) -> None:
        if self._current_image_path:
            self._save_current_annotations()
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return
        config = window.current_project_config
        if config is None:
            return
        if not self._annotations_dict and not self._image_list:
            return
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
                items.append(d)
            serialized[path] = items
        data = {"image_list": self._image_list, "annotations": serialized}
        annotations_path = ProjectManager.get_annotations_path(config)
        try:
            with open(annotations_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _load_annotations_from_project(self) -> None:
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return
        config = window.current_project_config
        if config is None:
            return
        annotations_path = ProjectManager.get_annotations_path(config)
        if not annotations_path.exists():
            return
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        image_list = data.get("image_list", [])
        annotations_data = data.get("annotations", {})
        valid_images = [p for p in image_list if os.path.isfile(p)]
        self._image_list = valid_images
        self._annotations_dict.clear()
        for path, items in annotations_data.items():
            if not os.path.isfile(path):
                continue
            anns: List[AnnotationItem] = []
            for d in items:
                rect = QRectF(*d["rect"]) if "rect" in d else QRectF()
                polygon = QPolygonF()
                if "polygon" in d:
                    for pt in d["polygon"]:
                        polygon.append(QPointF(pt[0], pt[1]))
                anns.append(AnnotationItem(
                    class_index=d.get("class_index", 0),
                    rect=rect,
                    polygon=polygon if "polygon" in d else QPolygonF(),
                    item_type=d.get("item_type", "rect"),
                ))
            self._annotations_dict[path] = anns
        self._image_list_widget.clear()
        for img_path in self._image_list:
            self._add_image_item(img_path)
