import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QImage, QIcon, QColor, QPainter, QBrush
from PyQt6.QtWidgets import (
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
)

from yolo26_app.core.annotation_canvas import AnnotationScene, AnnotationView, AnnotationItem
from yolo26_app.core.config import ClassItem, ProjectConfig
from yolo26_app.core.label_manager import LabelManager
from yolo26_app.core.yolo_exporter import YOLOExporter


class AnnotateWidget(QWidget):
    export_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._label_manager = LabelManager()
        self._annotations_dict: Dict[str, List[AnnotationItem]] = {}
        self._current_image_path: str = ""
        self._image_list: List[str] = []

        self._setup_ui()
        self._connect_signals()

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
        self._save_current_annotations()
        image_path = current.data(Qt.ItemDataRole.UserRole)
        self._current_image_path = image_path
        self._load_image(image_path)

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
        try:
            yaml_path, stats = YOLOExporter.export_dataset(
                self._annotations_dict, classes, output_dir
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
