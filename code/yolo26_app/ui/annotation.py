import json
import os
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QRectF, QPointF, QThread, QTimer
from PyQt6.QtGui import (
    QPixmap,
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
    QFileDialog,
    QFrame,
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
    QCheckBox,
    QDoubleSpinBox,
)

from yolo26_app.core.annotation_canvas import AnnotationScene, AnnotationView, AnnotationItem
from yolo26_app.core.config import ClassItem, ProjectConfig
from yolo26_app.core.label_manager import LabelManager
from yolo26_app.core.logger import get_logger
from yolo26_app.core.persistence import write_json_atomic

if TYPE_CHECKING:
    from yolo26_app.core.auto_annotator import SAMAnnotator
from yolo26_app.core.project_manager import ProjectManager
from yolo26_app.ui import styles

logger = get_logger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"}


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


class _YoloSamBatchWorker(QThread):
    """YOLO + SAM2 串联批量标注 worker:YOLO 预测 bbox → SAM2 用 box prompt 生成 mask → 转 polygon"""
    progress_signal = pyqtSignal(int, int)
    done_signal = pyqtSignal(dict, int)
    error_signal = pyqtSignal(str)

    def __init__(self, image_list, yolo_annotator, sam_predictor, conf):
        super().__init__()
        self._image_list = image_list
        self._yolo_annotator = yolo_annotator
        self._sam_predictor = sam_predictor
        self._conf = conf
        self._stop_flag = False

    def run(self):
        results_dict: Dict[str, List[AnnotationItem]] = {}
        total = len(self._image_list)
        # 显存优化:CUDA 可用时用 bfloat16 autocast 包裹 SAM2 调用
        use_autocast = False
        try:
            import torch
            use_autocast = torch.cuda.is_available()
        except ImportError:
            pass
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
        try:
            for i, img_path in enumerate(self._image_list):
                if self._stop_flag:
                    break
                try:
                    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        logger.warning(f"YOLO+SAM2 批量:图片读取失败,跳过 {img_path}")
                        continue
                    # YOLO 预测
                    predict_kwargs = dict(source=image, conf=self._conf, verbose=False)
                    model = self._yolo_annotator._model
                    results = model.predict(**predict_kwargs)
                    if not results:
                        continue
                    result = results[0]
                    if result.boxes is None or len(result.boxes) == 0:
                        continue
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                    classes = result.boxes.cls.cpu().numpy()  # (N,)
                    # SAM2 set_image 每图一次
                    with ctx:
                        self._sam_predictor.set_image(image)
                    anns: List[AnnotationItem] = []
                    h, w = image.shape[:2]
                    for box, cls in zip(boxes_xyxy, classes):
                        if self._stop_flag:
                            break
                        try:
                            with ctx:
                                masks, scores, logits = self._sam_predictor.predict(
                                    box=box,
                                    multimask_output=False,
                                )
                            mask = masks[0]
                            if mask is None:
                                continue
                            mask_u8 = (mask.astype(np.uint8)) * 255
                            # mask 尺寸可能与原图不一致(SAM2 内部 resize),对齐到原图
                            if mask_u8.shape[:2] != (h, w):
                                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
                            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not contours:
                                continue
                            # 取最大轮廓
                            contour = max(contours, key=cv2.contourArea)
                            # 简化 polygon
                            epsilon = 0.002 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            pts = approx.reshape(-1, 2)
                            if len(pts) < 3:
                                continue  # 至少 3 点才能构成 polygon
                            if len(pts) > 200:
                                logger.warning(f"YOLO+SAM2 批量:polygon 简化后点数 {len(pts)} > 200,跳过该标注 {img_path}")
                                continue
                            polygon = QPolygonF([QPointF(float(p[0]), float(p[1])) for p in pts])
                            anns.append(AnnotationItem(
                                class_index=int(cls),
                                polygon=polygon,
                                item_type="polygon",
                            ))
                        except Exception as e:
                            logger.warning(f"YOLO+SAM2 批量:单 bbox 处理失败,跳过 {img_path}: {e}")
                            continue
                    if anns:
                        results_dict[img_path] = anns
                except Exception as e:
                    logger.warning(f"YOLO+SAM2 批量:单图处理失败,跳过 {img_path}: {e}")
                    continue
                self.progress_signal.emit(i + 1, total)
            self.done_signal.emit(results_dict, total)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._stop_flag = True


class _DinoWorker(QThread):
    done_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, dino, image_path, text):
        super().__init__()
        self._dino = dino
        self._image_path = image_path
        self._text = text

    def run(self):
        try:
            annotations = self._dino.detect(self._image_path, self._text)
            self.done_signal.emit(annotations)
        except Exception as e:
            self.error_signal.emit(str(e))


class _ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(int, QPixmap)

    def __init__(self, items: list, parent=None):
        super().__init__(parent)
        self._items = items
        self._stop_flag = False

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        from PyQt6.QtGui import QImageReader
        for row, path in self._items:
            if self._stop_flag:
                break
            # 使用 QImageReader 提前设置缩放尺寸，避免全分辨率解码后再缩放
            reader = QImageReader(path)
            if reader.canRead():
                reader.setScaledSize(QSize(64, 64))
                img = reader.read()
                if not img.isNull():
                    pixmap = QPixmap.fromImage(img)
                    self.thumbnail_ready.emit(row, pixmap)


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
        self._yolo_sam_worker: Optional[_YoloSamBatchWorker] = None
        self._batch_progress = None
        self._dino_worker = None
        self._thumb_worker: Optional[_ThumbnailWorker] = None
        self._current_pixmap_item = None
        self._download_worker = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(1500)
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
            self._sam_worker.wait(5000)
            if self._sam_worker.isRunning():
                try:
                    self._sam_worker.disconnect()
                except (RuntimeError, TypeError):
                    pass
                logger.warning("警告: SAM encode worker 5 秒内未退出,已断开信号连接")
        import cv2
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
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
            self._sam_worker.wait(5000)
            if self._sam_worker.isRunning():
                try:
                    self._sam_worker.disconnect()
                except (RuntimeError, TypeError):
                    pass
                logger.warning("警告: SAM predict worker 5 秒内未退出,已断开信号连接")
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

    def _create_toolbar_separator(self) -> QFrame:
        """Create a vertical separator for toolbar grouping."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setObjectName("navSeparator")
        return sep

    def _load_icon(self, name: str) -> QIcon:
        """Load an SVG icon from the ui/icons directory by name."""
        icon_path = os.path.join(os.path.dirname(__file__), "icons", f"{name}.svg")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QIcon()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self._toolbar = QFrame()
        self._toolbar.setObjectName("annotateToolbar")
        self._toolbar.setFixedHeight(48)
        toolbar_layout = QHBoxLayout(self._toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(6)

        icon_size = QSize(16, 16)

        # --- Group 1: 标注工具 ---
        self._btn_rect = QPushButton("矩形标注")
        self._btn_rect.setCheckable(True)
        self._btn_rect.setChecked(True)
        self._btn_rect.setIcon(self._load_icon("tool-rect"))
        self._btn_rect.setIconSize(icon_size)
        self._btn_obb = QPushButton("OBB 旋转框")
        self._btn_obb.setCheckable(True)
        self._btn_obb.setToolTip("OBB 旋转框标注工具 (O)")
        self._btn_obb.setIcon(self._load_icon("tool-rect"))
        self._btn_obb.setIconSize(icon_size)
        self._btn_polygon = QPushButton("多边形标注")
        self._btn_polygon.setCheckable(True)
        self._btn_polygon.setIcon(self._load_icon("tool-polygon"))
        self._btn_polygon.setIconSize(icon_size)
        self._btn_keypoint = QPushButton("关键点")
        self._btn_keypoint.setCheckable(True)
        self._btn_keypoint.setToolTip("关键点标注工具")
        self._btn_keypoint.setIcon(self._load_icon("tool-keypoint"))
        self._btn_keypoint.setIconSize(icon_size)
        self._btn_select = QPushButton("选择")
        self._btn_select.setCheckable(True)
        self._btn_select.setIcon(self._load_icon("tool-select"))
        self._btn_select.setIconSize(icon_size)
        toolbar_layout.addWidget(self._btn_rect)
        toolbar_layout.addWidget(self._btn_obb)
        toolbar_layout.addWidget(self._btn_polygon)
        toolbar_layout.addWidget(self._btn_keypoint)
        toolbar_layout.addWidget(self._btn_select)

        toolbar_layout.addWidget(self._create_toolbar_separator())

        # --- Group 2: 编辑操作 ---
        self._btn_delete = QPushButton("删除")
        self._btn_delete.setIcon(self._load_icon("tool-delete"))
        self._btn_delete.setIconSize(icon_size)
        self._btn_clear_annotations = QPushButton("清空标注")
        self._btn_clear_annotations.setToolTip("清除当前画布的所有标注，保留图片本身")
        self._btn_clear_annotations.setIcon(self._load_icon("tool-clear"))
        self._btn_clear_annotations.setIconSize(icon_size)
        toolbar_layout.addWidget(self._btn_delete)
        toolbar_layout.addWidget(self._btn_clear_annotations)

        toolbar_layout.addWidget(self._create_toolbar_separator())

        # --- Group 3: AI辅助 ---
        self._btn_sam = QPushButton("SAM分割")
        self._btn_sam.setIcon(self._load_icon("tool-sam"))
        self._btn_sam.setIconSize(icon_size)
        self._btn_dino = QPushButton("文本检测")
        self._btn_dino.setIcon(self._load_icon("tool-dino"))
        self._btn_dino.setIconSize(icon_size)
        self._btn_batch = QPushButton("逐帧检测")
        self._btn_batch.setIcon(self._load_icon("tool-batch"))
        self._btn_batch.setIconSize(icon_size)
        toolbar_layout.addWidget(self._btn_sam)
        toolbar_layout.addWidget(self._btn_dino)
        toolbar_layout.addWidget(self._btn_batch)

        toolbar_layout.addWidget(self._create_toolbar_separator())

        # --- Group 4: 数据导入 ---
        self._btn_import_img = QPushButton("导入图片")
        self._btn_import_img.setIcon(self._load_icon("tool-import-image"))
        self._btn_import_img.setIconSize(icon_size)
        self._btn_import_video = QPushButton("导入视频")
        self._btn_import_video.setIcon(self._load_icon("tool-import-video"))
        self._btn_import_video.setIconSize(icon_size)
        self._btn_import_dir = QPushButton("导入目录")
        self._btn_import_dir.setIcon(self._load_icon("tool-import-dir"))
        self._btn_import_dir.setIconSize(icon_size)
        self._btn_clear_images = QPushButton("清空图片")
        self._btn_clear_images.setToolTip("清空当前标注区的图片列表和对应标注，不删除磁盘文件")
        self._btn_clear_images.setIcon(self._load_icon("tool-clear"))
        self._btn_clear_images.setIconSize(icon_size)
        toolbar_layout.addWidget(self._btn_import_img)
        toolbar_layout.addWidget(self._btn_import_video)
        toolbar_layout.addWidget(self._btn_import_dir)
        toolbar_layout.addWidget(self._btn_clear_images)

        toolbar_layout.addWidget(self._create_toolbar_separator())

        # --- Group 5: 导出 ---
        self._btn_export = QPushButton("导出数据集")
        self._btn_export.setIcon(self._load_icon("tool-export"))
        self._btn_export.setIconSize(icon_size)
        toolbar_layout.addWidget(self._btn_export)

        toolbar_layout.addStretch()
        self._toolbar.setStyleSheet(styles.TOOLBAR_BUTTON_STYLE)

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
        self._btn_add_class = QPushButton()
        self._btn_add_class.setIcon(self._load_icon("action-add"))
        self._btn_add_class.setIconSize(QSize(14, 14))
        self._btn_add_class.setMinimumWidth(40)
        self._btn_remove_class = QPushButton()
        self._btn_remove_class.setIcon(self._load_icon("action-remove"))
        self._btn_remove_class.setIconSize(QSize(14, 14))
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
        self._splitter.setSizes([280, 720])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._splitter)
        self.setLayout(main_layout)

    def _connect_signals(self) -> None:
        self._btn_rect.clicked.connect(lambda: self._set_tool("rect"))
        self._btn_obb.clicked.connect(lambda: self._set_tool("obb"))
        self._btn_polygon.clicked.connect(lambda: self._set_tool("polygon"))
        self._btn_keypoint.clicked.connect(lambda: self._set_tool("keypoint"))
        self._btn_select.clicked.connect(lambda: self._set_tool("select"))
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_clear_annotations.clicked.connect(self._clear_current_annotations)
        self._btn_sam.clicked.connect(self._sam_annotate)
        self._btn_dino.clicked.connect(self._text_detect)
        self._btn_batch.clicked.connect(self._batch_detect)
        self._btn_import_img.clicked.connect(self._import_images)
        self._btn_import_video.clicked.connect(self._import_video)
        self._btn_import_dir.clicked.connect(self._import_directory)
        self._btn_clear_images.clicked.connect(self._clear_imported_images)
        self._btn_export.clicked.connect(self._export_dataset)
        self._image_list_widget.currentItemChanged.connect(self._on_image_selected)
        self._btn_add_class.clicked.connect(self._add_class)
        self._btn_remove_class.clicked.connect(self._remove_class)
        self._class_list_widget.currentRowChanged.connect(self._on_class_selected)
        self._scene.annotations_changed.connect(self._on_annotations_changed)

        self._next_image_shortcut = QShortcut(QKeySequence("Shift+Space"), self)
        self._next_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_image_shortcut.setAutoRepeat(False)
        self._next_image_shortcut.activated.connect(self._go_to_next_image)

        self._delete_shortcut = QShortcut(QKeySequence("Space"), self)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.setAutoRepeat(False)
        self._delete_shortcut.activated.connect(self._delete_selected)

        # 工具切换快捷键:R=矩形 / P=多边形 / O=OBB / K=关键点 / S=选择
        self._rect_tool_shortcut = QShortcut(QKeySequence("R"), self)
        self._rect_tool_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._rect_tool_shortcut.activated.connect(lambda: self._set_tool("rect"))

        self._polygon_tool_shortcut = QShortcut(QKeySequence("P"), self)
        self._polygon_tool_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._polygon_tool_shortcut.activated.connect(lambda: self._set_tool("polygon"))

        self._obb_tool_shortcut = QShortcut(QKeySequence("O"), self)
        self._obb_tool_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._obb_tool_shortcut.activated.connect(lambda: self._set_tool("obb"))

        self._keypoint_tool_shortcut = QShortcut(QKeySequence("K"), self)
        self._keypoint_tool_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._keypoint_tool_shortcut.activated.connect(lambda: self._set_tool("keypoint"))

        self._select_tool_shortcut = QShortcut(QKeySequence("S"), self)
        self._select_tool_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._select_tool_shortcut.activated.connect(lambda: self._set_tool("select"))

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

    def stop_background_threads(self) -> None:
        """停止所有后台线程（缩略图、SAM、批量检测等），用于窗口关闭前清理。

        先对所有 worker 发出停止信号，再并行等待，避免串行等待导致的超时叠加。
        """
        self._stop_thumb_worker()
        # 收集所有需要停止的 worker
        workers = []
        if self._sam_worker is not None:
            workers.append(("SAM", self._sam_worker))
        if self._batch_worker is not None:
            self._batch_worker.stop()
            workers.append(("批量检测", self._batch_worker))
        if self._yolo_sam_worker is not None:
            self._yolo_sam_worker.stop()
            workers.append(("YOLO+SAM2", self._yolo_sam_worker))
        # 并行等待：先对所有 worker 发出 stop，再统一等待
        for name, w in workers:
            try:
                if w.isRunning():
                    w.wait(3000)
                    if w.isRunning():
                        try:
                            w.disconnect()
                        except (RuntimeError, TypeError):
                            pass
                        logger.warning("警告: %s worker 3 秒内未退出,已断开信号连接", name)
            except RuntimeError:
                pass
        self._sam_worker = None
        self._batch_worker = None
        self._yolo_sam_worker = None

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
        self._btn_obb.setChecked(tool == "obb")
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

    def _clear_current_annotations(self) -> None:
        if not self._current_image_path:
            QMessageBox.information(self, "提示", "当前没有选中的图片")
            return
        if not self._scene.get_annotations():
            QMessageBox.information(self, "提示", "当前画布没有可清除的标注")
            return

        reply = QMessageBox.question(
            self,
            "确认清空标注",
            "将清除当前画布的所有标注（保留图片本身）。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._scene.clear_annotations()
        self._annotations_dict[self._current_image_path] = []
        self._schedule_autosave()

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
            return os.path.join(project_path, rel)

        # 复制到 images/，重名加序号
        dest = images_dir / src.name
        if dest.resolve() == src.resolve():
            return str(dest.resolve())
        counter = 1
        stem, suffix = src.stem, src.suffix
        while dest.exists():
            dest = images_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        import shutil
        shutil.copy2(str(src), str(dest))
        return str(dest.resolve())

    def _import_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "导入图片",
            self._get_import_start_dir(),
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)",
        )
        if not files:
            return
        added = self._import_image_files(files)
        if added:
            self._finish_media_import()

    def _import_directory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "选择媒体目录", self._get_import_start_dir())
        if not dir_path:
            return

        image_files, video_files = self._scan_media_directory(dir_path)
        if not image_files and not video_files:
            QMessageBox.warning(self, "提示", "目录中未找到支持的图片或视频文件")
            return

        added_images = self._import_image_files(image_files)
        imported_videos, extracted_frames, failed_videos = self._import_video_files(video_files)
        if added_images or extracted_frames:
            self._finish_media_import(select_first_if_empty=True)

        summary = f"图片 {added_images} 张，视频 {imported_videos} 个，抽帧 {extracted_frames} 张"
        if failed_videos:
            summary += "\n以下视频无法打开或没有有效帧，已跳过：\n" + "\n".join(failed_videos[:10])
            if len(failed_videos) > 10:
                summary += f"\n... 另有 {len(failed_videos) - 10} 个"
            QMessageBox.warning(self, "导入完成", summary)
        else:
            QMessageBox.information(self, "导入完成", summary)

    def _import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入视频", self._get_import_start_dir(), "Videos (*.mp4 *.avi *.mkv *.mov *.wmv *.flv)"
        )
        if not path:
            return

        _imported_videos, extracted_frames, failed_videos = self._import_video_files([path])
        if extracted_frames:
            self._finish_media_import()
        if failed_videos:
            QMessageBox.warning(
                self,
                "导入视频失败",
                "视频无法打开或没有有效帧，已跳过：\n" + "\n".join(failed_videos),
            )

    def _scan_media_directory(self, dir_path: str) -> tuple[List[str], List[str]]:
        image_files: List[str] = []
        video_files: List[str] = []
        for root, dirs, files in os.walk(dir_path):
            dirs.sort()
            for filename in sorted(files):
                path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(path)
                elif ext in VIDEO_EXTENSIONS:
                    video_files.append(path)
        image_files.sort(key=str.lower)
        video_files.sort(key=str.lower)
        return image_files, video_files

    def _import_image_files(self, paths: List[str]) -> int:
        added = 0
        total = len(paths)
        show_progress = total > 50
        if show_progress:
            progress = QProgressDialog(f"正在导入 0/{total}...", "取消", 0, total, self)
            progress.setWindowTitle("导入图片")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
        for i, path in enumerate(paths):
            if path and os.path.isfile(path):
                stored = self._copy_to_project_images(path)
                if self._add_imported_image(stored):
                    added += 1
            if show_progress:
                progress.setValue(i + 1)
                progress.setLabelText(f"正在导入 {i + 1}/{total}...")
                if progress.wasCanceled():
                    break
        if show_progress:
            progress.setValue(total)
        return added

    def _import_video_files(self, paths: List[str]) -> tuple[int, int, List[str]]:
        imported_videos = 0
        extracted_frames = 0
        failed_videos: List[str] = []

        for path in paths:
            if not path or not os.path.isfile(path):
                failed_videos.append(os.path.basename(path) if path else "未知视频")
                continue
            try:
                frame_paths = self._extract_video_frames(path)
            except Exception as exc:
                failed_videos.append(f"{os.path.basename(path)}: {exc}")
                continue

            added_for_video = 0
            for stored in frame_paths:
                if self._add_imported_image(stored):
                    added_for_video += 1
            if added_for_video:
                imported_videos += 1
                extracted_frames += added_for_video
            else:
                failed_videos.append(os.path.basename(path))

        return imported_videos, extracted_frames, failed_videos

    def _extract_video_frames(self, video_path: str) -> List[str]:
        import cv2

        output_dir = self._get_video_frame_output_dir()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        interval = max(1, int(fps))
        frame_idx = 0
        saved_idx = 0
        stored_paths: List[str] = []
        has_project = self._current_project_config() is not None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_path = self._next_video_frame_path(output_dir, video_path, saved_idx)
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    buf.tofile(str(out_path))
                    stored = self._copy_to_project_images(str(out_path)) if has_project else str(out_path)
                    stored_paths.append(stored)
                    saved_idx += 1
            frame_idx += 1

        cap.release()
        return stored_paths

    def _current_project_config(self) -> Optional[ProjectConfig]:
        window = self.window()
        if hasattr(window, "current_project_config"):
            return window.current_project_config
        return None

    def _get_import_start_dir(self) -> str:
        """返回导入对话框的起始目录。

        优先使用当前工作区间的 images/ 目录(若已存在),
        否则回退到 project_path,再否则回退到用户主目录。
        """
        config = self._current_project_config()
        if config is not None:
            images_dir = ProjectManager.get_images_dir(config)
            if images_dir.is_dir():
                return str(images_dir)
            if Path(config.project_path).is_dir():
                return str(config.project_path)
        return str(Path.home())

    def _get_video_frame_output_dir(self) -> Path:
        config = self._current_project_config()
        if config is not None:
            output_dir = ProjectManager.get_images_dir(config)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="yolo26_frames_"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _next_video_frame_path(self, output_dir: Path, video_path: str, frame_index: int) -> Path:
        stem = Path(video_path).stem
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem).strip("._")
        if not safe_stem:
            safe_stem = "video"
        candidate = output_dir / f"{safe_stem}_frame_{frame_index:06d}.jpg"
        counter = 1
        while candidate.exists():
            candidate = output_dir / f"{safe_stem}_frame_{frame_index:06d}_{counter}.jpg"
            counter += 1
        return candidate

    def _add_imported_image(self, image_path: str) -> bool:
        if image_path in self._image_list:
            return False
        self._image_list.append(image_path)
        self._annotations_dict.setdefault(image_path, [])
        self._add_image_item(image_path)
        return True

    def _finish_media_import(self, select_first_if_empty: bool = False) -> None:
        self._start_thumbnail_loading()
        if select_first_if_empty and self._image_list_widget.currentRow() < 0 and self._image_list_widget.count() > 0:
            self._image_list_widget.setCurrentRow(0)
        self._schedule_autosave()

    def _add_image_item(self, image_path: str) -> None:
        item = QListWidgetItem(os.path.basename(image_path))
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        item.setToolTip(image_path)
        self._image_list_widget.addItem(item)

    def _stop_thumb_worker(self) -> None:
        """停止当前缩略图 worker 并完整等待其退出。

        多图场景下旧 worker 可能仍在处理大图，必须等待 run() 自然退出，
        否则后续覆盖引用会导致 QThread 被提前销毁。
        """
        worker = self._thumb_worker
        self._thumb_worker = None
        if worker is None:
            return
        try:
            worker.stop()
            if worker.isRunning():
                worker.wait(3000)
        except RuntimeError:
            pass

    def _start_thumbnail_loading(self) -> None:
        items = []
        for row in range(self._image_list_widget.count()):
            item = self._image_list_widget.item(row)
            path = item.data(Qt.ItemDataRole.UserRole)
            items.append((row, path))
        if not items:
            self._stop_thumb_worker()
            return

        # 先完整停止旧 worker，避免多个线程并发运行
        self._stop_thumb_worker()

        # parent=None，避免 widget 销毁时 Qt 自动销毁仍在运行的线程
        worker = _ThumbnailWorker(items, None)
        worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        worker.finished.connect(worker.deleteLater)
        # 用局部变量比较，避免旧 worker 的 finished 误清新 worker 引用
        def _on_finished(_w=worker):
            if self._thumb_worker is _w:
                self._thumb_worker = None
        worker.finished.connect(_on_finished)
        self._thumb_worker = worker
        worker.start()

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
        # clear_annotations 已移除旧 pixmap item，重置引用避免悬空
        self._current_pixmap_item = None
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return
        if self._current_pixmap_item is not None:
            self._scene.removeItem(self._current_pixmap_item)
        self._current_pixmap_item = self._scene.addPixmap(pixmap)
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

    def _clear_imported_images(self) -> None:
        if not self._image_list:
            QMessageBox.information(self, "提示", "当前标注区没有可清空的图片")
            return

        reply = QMessageBox.question(
            self,
            "确认清空",
            "将清空当前标注区的所有图片和对应标注记录。\n不会删除磁盘上的图片文件。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self._thumb_worker is not None:
            self._stop_thumb_worker()

        self._save_current_annotations()
        self._image_list.clear()
        self._annotations_dict.clear()
        self._current_image_path = ""
        self._image_list_widget.clear()
        self._scene.clear_annotations()
        self._current_pixmap_item = None
        self._scene.clear()
        self._scene.setSceneRect(QRectF())
        self._save_annotations_to_project(force=True)
        self._schedule_autosave()

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
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush(QColor(cls_item.color)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(1, 1, 14, 14)
            painter.end()
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
            # device 需在创建下载 worker 之前计算，确保下载完成回调 lambda 可引用
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
                filename = os.path.basename(model_path).lower()
                if "hiera_l" in filename or "hiera-large" in filename:
                    config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
                elif "hiera_b" in filename or "hiera-base" in filename:
                    config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
                elif "hiera_t" in filename or "hiera-tiny" in filename:
                    config_path = "configs/sam2.1/sam2.1_hiera_t.yaml"
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
        self._dino_worker = _DinoWorker(dino, self._current_image_path, text.strip())
        self._dino_worker.done_signal.connect(self._on_dino_done)
        self._dino_worker.error_signal.connect(
            lambda msg: QMessageBox.warning(self, "检测失败", msg)
        )
        self._dino_worker.finished.connect(self._dino_worker.deleteLater)
        self._dino_worker.finished.connect(lambda: setattr(self, '_dino_worker', None))
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage("正在检测...")
        self._dino_worker.start()

    def _on_dino_done(self, annotations) -> None:
        self._scene.load_annotations(annotations)
        self._annotations_dict[self._current_image_path] = annotations
        self._schedule_autosave()
        window = self.window()
        if hasattr(window, "statusbar"):
            window.statusbar.showMessage("就绪")
        if not annotations:
            QMessageBox.information(self, "提示", "未检测到目标")

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
        # 自定义对话框:置信度 + SAM2 复选框
        dialog = QDialog(self)
        dialog.setWindowTitle("批量检测")
        form = QFormLayout(dialog)
        conf_spin = QDoubleSpinBox(dialog)
        conf_spin.setRange(0.01, 1.0)
        conf_spin.setSingleStep(0.05)
        conf_spin.setValue(0.25)
        conf_spin.setDecimals(2)
        sam_check = QCheckBox("使用 SAM2 生成精确掩码(polygon)", dialog)
        sam_check.setChecked(False)
        form.addRow("置信度阈值:", conf_spin)
        form.addRow(sam_check)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        form.addRow(button_box)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        conf = conf_spin.value()
        use_sam = sam_check.isChecked()
        # SAM2 就绪性校验
        sam_predictor = None
        if use_sam:
            sam_annotator = self._get_sam_annotator()
            if not sam_annotator.available:
                QMessageBox.warning(self, "SAM2 未安装", "请先安装 SAM 2:\n  pip install sam2\n并在标注区加载 SAM2 模型")
                return
            if sam_annotator._predictor is None:
                QMessageBox.warning(self, "SAM2 未加载", "请先在标注区点击 SAM 分割按钮加载 SAM2 模型")
                return
            sam_predictor = sam_annotator._predictor
        total = len(self._image_list)
        progress = QProgressDialog("正在处理图片...", "取消", 0, total, self)
        progress.setWindowTitle("YOLO+SAM2 批量标注" if use_sam else "逐帧检测")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        self._batch_progress = progress
        if use_sam:
            self._yolo_sam_worker = _YoloSamBatchWorker(self._image_list, yolo, sam_predictor, conf)
            worker = self._yolo_sam_worker
        else:
            self._batch_worker = _BatchDetectWorker(self._image_list, yolo, conf)
            worker = self._batch_worker
        worker.progress_signal.connect(progress.setValue)
        worker.done_signal.connect(self._on_batch_done)
        worker.error_signal.connect(
            lambda msg: QMessageBox.critical(self, "错误", msg)
        )
        progress.canceled.connect(worker.stop)
        worker.start()
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._on_batch_worker_finished)

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

    def _on_batch_worker_finished(self) -> None:
        """批量检测 worker 完成时清理引用,避免 Python/C++ 对象生命周期不同步。"""
        self._batch_worker = None
        self._yolo_sam_worker = None

    def _generate_default_dataset_name(self, datasets_dir: Path) -> str:
        """扫描 datasets 目录,生成不冲突的 dataset1/dataset2/... 默认名称。"""
        i = 1
        while (datasets_dir / f"dataset{i}").exists():
            i += 1
        return f"dataset{i}"

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
            datasets_dir = ProjectManager.get_dataset_dir(window.current_project_config)
            datasets_dir.mkdir(parents=True, exist_ok=True)
            default_name = self._generate_default_dataset_name(datasets_dir)
            name, ok = QInputDialog.getText(
                self, "导出数据集", "请输入数据集名称:", QLineEdit.Normal, default_name
            )
            if not ok or not name.strip():
                return
            output_dir = str(datasets_dir / name.strip())
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
        has_obb = any(
            a.item_type == "obb"
            for anns in self._annotations_dict.values()
            for a in anns
        )
        task = "detect"
        if has_polygon or has_obb:
            items = [
                "detect — 多边形自动转为矩形框",
                "segment — 保留多边形用于分割训练",
                "pose — 关键点姿态格式",
                "obb — 旋转框格式 (cx cy w h angle)",
                "classify — 分类任务（使用目录结构）",
            ]
            item, ok = QInputDialog.getItem(
                self, "选择导出格式",
                "检测到多边形或 OBB 标注，请选择导出格式：",
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
            elif "obb" in item:
                task = "obb"
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

    def set_project_config(self, config: Optional[ProjectConfig]) -> None:
        self._autosave_timer.stop()
        self._current_image_path = ""
        self._image_list.clear()
        self._annotations_dict.clear()
        self._image_list_widget.clear()
        self._scene.clear_annotations()
        if config is None:
            # 自由空间模式:不加载项目数据
            return
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
                elif ann.item_type == "obb" and ann.rect is not None:
                    # OBB 旋转框:存储外接矩形 + angle(弧度)
                    d["rect"] = [ann.rect.x(), ann.rect.y(), ann.rect.width(), ann.rect.height()]
                    d["angle"] = ann.angle
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
                # OBB 类型读取 angle(弧度),向后兼容老数据默认 0.0
                angle = d.get("angle", 0.0) if d.get("item_type") == "obb" else 0.0
                anns.append(AnnotationItem(
                    class_index=d.get("class_index", 0),
                    rect=rect,
                    polygon=polygon if "polygon" in d else QPolygonF(),
                    item_type=d.get("item_type", "rect"),
                    keypoints=keypoints,
                    angle=angle,
                ))
            self._annotations_dict[path] = anns
        self._image_list_widget.clear()
        for img_path in self._image_list:
            self._add_image_item(img_path)
        self._start_thumbnail_loading()
        current_path = state.get("current_image_path", "")
        if current_path in self._image_list:
            self._image_list_widget.setCurrentRow(self._image_list.index(current_path))

    def _save_annotations_to_project(self, force: bool = False) -> bool:
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return False
        config = window.current_project_config
        if config is None:
            return False
        if not force and not self._annotations_dict and not self._image_list:
            return False
        project_path = config.project_path
        serialized: Dict[str, list] = {}
        for path, anns in self._annotations_dict.items():
            items = []
            for ann in anns:
                d = {"class_index": ann.class_index, "item_type": ann.item_type}
                if ann.item_type == "rect" and ann.rect is not None:
                    d["rect"] = [ann.rect.x(), ann.rect.y(), ann.rect.width(), ann.rect.height()]
                elif ann.item_type == "obb" and ann.rect is not None:
                    # OBB 旋转框:存储外接矩形 + angle(弧度)
                    d["rect"] = [ann.rect.x(), ann.rect.y(), ann.rect.width(), ann.rect.height()]
                    d["angle"] = ann.angle
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
        # 备份上一版 annotations.json 为 annotations.bak.json
        try:
            if annotations_path.exists():
                bak_path = annotations_path.with_name("annotations.bak.json")
                shutil.copy2(annotations_path, bak_path)
        except (PermissionError, OSError) as e:
            logger.warning(f"备份 annotations.bak.json 失败: {e}")
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
        # 批量收集所有需要检查的路径，一次性去重后用 os.scandir 批量验证，
        # 避免 N+M 次独立 os.path.isfile 调用（每次都是独立系统调用）。
        all_paths_to_check = set(image_list)
        for path in annotations_data.keys():
            if os.path.isabs(path):
                all_paths_to_check.add(path)
            else:
                all_paths_to_check.add(os.path.join(project_path, path))
        existing_files: set = set()
        if all_paths_to_check:
            # 按父目录分组，用 os.scandir 一次列出目录内容
            dir_groups: dict = {}
            for p in all_paths_to_check:
                parent = os.path.dirname(p)
                dir_groups.setdefault(parent, []).append(os.path.basename(p))
            for dir_path, basenames in dir_groups.items():
                try:
                    existing_in_dir = {entry.name for entry in os.scandir(dir_path) if entry.is_file()}
                    for bn in basenames:
                        if bn in existing_in_dir:
                            existing_files.add(os.path.join(dir_path, bn))
                except OSError:
                    pass
        valid_images = [p for p in image_list if p in existing_files]
        self._image_list = valid_images
        self._annotations_dict.clear()
        for path, items in annotations_data.items():
            if os.path.isabs(path):
                abs_path = path
            else:
                abs_path = os.path.join(project_path, path)
            if abs_path not in existing_files:
                continue
            anns: List[AnnotationItem] = []
            for d in items:
                rect = QRectF(*d["rect"]) if "rect" in d else QRectF()
                polygon = QPolygonF()
                if "polygon" in d:
                    for pt in d["polygon"]:
                        polygon.append(QPointF(pt[0], pt[1]))
                keypoints = [QPointF(pt[0], pt[1]) for pt in d.get("keypoints", [])]
                # OBB 类型读取 angle(弧度),向后兼容老数据默认 0.0
                angle = d.get("angle", 0.0) if d.get("item_type") == "obb" else 0.0
                anns.append(AnnotationItem(
                    class_index=d.get("class_index", 0),
                    rect=rect,
                    polygon=polygon if "polygon" in d else QPolygonF(),
                    item_type=d.get("item_type", "rect"),
                    keypoints=keypoints,
                    angle=angle,
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
