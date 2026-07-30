import os
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QObject, QThread, QWaitCondition, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from yolo26_app.core.config import ProjectConfig
from yolo26_app.core.logger import get_logger
from yolo26_app.core.predictor import YOLOPredictor
from yolo26_app.core.realsense_camera import RealSenseCamera
from yolo26_app.ui import styles
from yolo26_app.ui.export_dialog import ExportDialog

logger = get_logger(__name__)


class _ValidateWorker(QThread):
    done_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, predictor: YOLOPredictor, data_path: str) -> None:
        super().__init__()
        self.predictor = predictor
        self.data_path = data_path
        self._stop_flag = False

    def request_stop(self) -> None:
        """设置停止标志，由 run() 协作检查。"""
        self._stop_flag = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_flag

    def run(self) -> None:
        if self._stop_flag:
            self.error_signal.emit("任务已取消")
            return
        try:
            result = self.predictor.validate_model(self.data_path)
            if self._stop_flag:
                self.error_signal.emit("任务已取消")
            else:
                self.done_signal.emit(result)
        except Exception as e:
            if not self._stop_flag:
                self.error_signal.emit(str(e))


class _ExportWorker(QThread):
    done_signal = pyqtSignal(str, bool, str)  # path, success, error_msg
    error_signal = pyqtSignal(str)

    def __init__(self, predictor: YOLOPredictor, format: str, output_dir: str, **kwargs) -> None:
        super().__init__()
        self.predictor = predictor
        self.format = format
        self.output_dir = output_dir
        self.kwargs = kwargs
        self._stop_flag = False

    def request_stop(self) -> None:
        """设置停止标志，由 run() 协作检查。"""
        self._stop_flag = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_flag

    def run(self) -> None:
        if self._stop_flag:
            self.error_signal.emit("任务已取消")
            return
        try:
            exported_path, success, error_msg = self.predictor.export_model(self.format, self.output_dir, **self.kwargs)
            if self._stop_flag:
                self.error_signal.emit("任务已取消")
            else:
                self.done_signal.emit(exported_path, success, error_msg)
        except Exception as e:
            if not self._stop_flag:
                self.error_signal.emit(str(e))


class _ImagePredictWorker(QThread):
    done_signal = pyqtSignal(np.ndarray, object)
    error_signal = pyqtSignal(str)

    def __init__(self, predictor, image_path: str, conf: float, iou: float, imgsz: int = 640, device: str = "", max_det: int = 300) -> None:
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self._stop_flag = False

    def request_stop(self) -> None:
        """设置停止标志，由 run() 协作检查。"""
        self._stop_flag = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_flag

    def run(self) -> None:
        if self._stop_flag:
            self.error_signal.emit("任务已取消")
            return
        try:
            annotated, results = self.predictor.predict_image(
                self.image_path, conf=self.conf, iou=self.iou,
                imgsz=self.imgsz, device=self.device, max_det=self.max_det
            )
            if self._stop_flag:
                self.error_signal.emit("任务已取消")
            else:
                self.done_signal.emit(annotated, results)
        except Exception as e:
            if not self._stop_flag:
                self.error_signal.emit(str(e))


class _InferenceWorker(QThread):
    result_signal = pyqtSignal(np.ndarray, object)

    def __init__(self, predictor: YOLOPredictor, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._predictor = predictor
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._frame: Optional[np.ndarray] = None
        self._conf: float = 0.25
        self._iou: float = 0.7
        self._imgsz: int = 640
        self._device: str = ""
        self._max_det: int = 300
        self._busy: bool = False
        self._stop_flag: bool = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def submit(self, frame: np.ndarray, conf: float, iou: float, imgsz: int = 640, device: str = "", max_det: int = 300) -> None:
        self._mutex.lock()
        self._frame = frame.copy()
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._max_det = max_det
        self._busy = True
        self._cond.wakeOne()
        self._mutex.unlock()

    def stop(self) -> None:
        """请求停止线程，不阻塞 UI 线程，不重置 _stop_flag。

        线程真正退出后由 finished 信号触发清理。
        若需同步等待（如 closeEvent），调用方应自行 wait()。
        """
        self._mutex.lock()
        self._stop_flag = True
        self._cond.wakeOne()
        self._mutex.unlock()

    def run(self) -> None:
        while True:
            self._mutex.lock()
            while self._frame is None and not self._stop_flag:
                self._cond.wait(self._mutex)
            if self._stop_flag:
                self._mutex.unlock()
                break
            frame = self._frame
            conf = self._conf
            iou = self._iou
            imgsz = self._imgsz
            device = self._device
            max_det = self._max_det
            self._frame = None
            self._mutex.unlock()

            try:
                annotated, results = self._predictor.predict_frame(
                    frame, conf=conf, iou=iou, imgsz=imgsz, device=device, max_det=max_det
                )
            except Exception:
                annotated = frame
                results = None
            self.result_signal.emit(annotated, results)

            self._mutex.lock()
            self._busy = False
            self._mutex.unlock()


class _ModelLoadWorker(QThread):
    """后台加载模型，避免 ONNX warmup / CUDA 初始化阻塞 UI。"""
    finished_signal = pyqtSignal(bool, str, dict)  # success, model_path, model_info

    def __init__(self, predictor: YOLOPredictor, model_path: str, task: str = "") -> None:
        super().__init__()
        self._predictor = predictor
        self._model_path = model_path
        self._task = task

    def run(self) -> None:
        try:
            success = self._predictor.load_model(self._model_path, task=self._task)
            info = self._predictor.get_model_info() if success else {}
            self.finished_signal.emit(success, self._model_path, info)
        except Exception:
            self.finished_signal.emit(False, self._model_path, {})


class TestWidget(QWidget):
    model_loaded = pyqtSignal(object)

    # 模型任务状态机
    STATE_IDLE = "idle"
    STATE_LOADING = "loading"
    STATE_PREDICTING = "predicting"
    STATE_VALIDATING = "validating"
    STATE_EXPORTING = "exporting"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.predictor = YOLOPredictor()
        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_timeout)
        self._last_frame_time: float = 0.0
        self._fps: float = 0.0
        self._project_config: Optional[ProjectConfig] = None
        self.rs_camera = RealSenseCamera()
        self._show_depth = False
        self._batch_images: List[str] = []
        self._batch_index: int = 0
        self._inference_worker = _InferenceWorker(self.predictor, parent=None)
        self._inference_worker.result_signal.connect(self._on_inference_result)
        self._last_frame: Optional[np.ndarray] = None
        self._image_predict_worker: Optional[_ImagePredictWorker] = None
        self._model_load_worker: Optional[_ModelLoadWorker] = None
        self._validate_worker: Optional[_ValidateWorker] = None
        self._export_worker: Optional[_ExportWorker] = None
        self._model_state: str = self.STATE_IDLE
        self._init_ui()

    def _init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: controls (minimum 360px, scrollable, resizable via splitter)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(360)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        # Right panel: results (elastic)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 12, 12, 12)
        right_layout.setSpacing(8)

        model_group = QGroupBox("模型设置")
        model_group.setObjectName("configCard")
        model_form = QFormLayout()
        model_form.setSpacing(4)
        model_group.setLayout(model_form)

        path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择模型文件 (.pt/.onnx/.torchscript/.xml/.engine)")
        path_layout.addWidget(self.model_path_edit)
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self._on_browse_model)
        path_layout.addWidget(self.browse_btn)
        model_form.addRow("模型路径:", path_layout)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        model_form.addRow("置信度阈值:", self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.7)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        model_form.addRow("IoU阈值:", self.iou_spin)

        # 高级预测参数（可折叠）
        self._advanced_widgets: list = []
        self._advanced_toggle_btn = QPushButton("高级参数 ▼")
        self._advanced_toggle_btn.setCheckable(True)
        self._advanced_toggle_btn.setChecked(False)
        self._advanced_toggle_btn.clicked.connect(self._toggle_advanced_params)
        model_form.addRow(self._advanced_toggle_btn)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        model_form.addRow("推理尺寸 (imgsz):", self.imgsz_spin)
        self._advanced_widgets.append(self.imgsz_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItem("自动", "")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU (0)", "0")
        model_form.addRow("设备 (device):", self.device_combo)
        self._advanced_widgets.append(self.device_combo)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)
        self.max_det_spin.setValue(300)
        model_form.addRow("最大检测数 (max_det):", self.max_det_spin)
        self._advanced_widgets.append(self.max_det_spin)

        # 初始隐藏高级参数控件
        for w in self._advanced_widgets:
            w.setVisible(False)

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self._on_load_model)
        model_form.addRow(self.load_model_btn)

        self.model_state_label = QLabel("状态: 空闲")
        model_form.addRow(self.model_state_label)

        input_group = QGroupBox("推理输入")
        input_group.setObjectName("configCard")
        input_layout = QHBoxLayout()
        input_group.setLayout(input_layout)

        self.image_btn = QPushButton("选择图片")
        self.image_btn.clicked.connect(self._on_select_image)
        input_layout.addWidget(self.image_btn)

        self.dir_btn = QPushButton("选择图片目录")
        self.dir_btn.clicked.connect(self._select_image_directory)
        input_layout.addWidget(self.dir_btn)

        self.video_btn = QPushButton("选择视频")
        self.video_btn.clicked.connect(self._on_select_video)
        input_layout.addWidget(self.video_btn)

        self.camera_btn = QPushButton("打开摄像头")
        self.camera_btn.clicked.connect(self._on_open_camera)
        input_layout.addWidget(self.camera_btn)

        realsense_group = QGroupBox("RealSense")
        realsense_group.setObjectName("configCard")
        rs_layout = QVBoxLayout(realsense_group)
        rs_layout.setSpacing(4)

        self.rs_device_combo = QComboBox()
        self.rs_device_combo.setPlaceholderText("选择 RealSense 设备")
        self.rs_device_combo.setMinimumWidth(160)
        rs_layout.addWidget(self.rs_device_combo)

        self.rs_refresh_btn = QPushButton("刷新设备")
        self.rs_refresh_btn.clicked.connect(self._on_refresh_rs_devices)
        rs_layout.addWidget(self.rs_refresh_btn)

        self.rs_camera_btn = QPushButton("打开 RealSense")
        self.rs_camera_btn.clicked.connect(self._on_open_rs_camera)
        rs_layout.addWidget(self.rs_camera_btn)

        self.depth_check = QCheckBox("显示深度图")
        self.depth_check.toggled.connect(self._on_depth_check_toggled)
        rs_layout.addWidget(self.depth_check)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        rs_layout.addWidget(self.stop_btn)

        self.result_label = QLabel("等待推理...")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.result_label.setStyleSheet(styles.RESULT_LABEL_STYLE)

        results_group = QGroupBox("结果与指标")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(4)
        results_group.setLayout(results_layout)

        stats_layout = QHBoxLayout()
        self.det_count_label = QLabel("检测数量: 0")
        stats_layout.addWidget(self.det_count_label)
        self.fps_label = QLabel("FPS: 0")
        stats_layout.addWidget(self.fps_label)
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self._prev_batch_image)
        self.prev_btn.setVisible(False)
        stats_layout.addWidget(self.prev_btn)
        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self._next_batch_image)
        self.next_btn.setVisible(False)
        stats_layout.addWidget(self.next_btn)
        self.validate_btn = QPushButton("验证模型")
        self.validate_btn.clicked.connect(self._on_validate)
        stats_layout.addWidget(self.validate_btn)
        self.export_btn = QPushButton("导出模型")
        self.export_btn.clicked.connect(self._on_export_clicked)
        stats_layout.addWidget(self.export_btn)
        results_layout.addLayout(stats_layout)

        self.val_result_group = QGroupBox("验证结果")
        val_layout = QVBoxLayout()
        self.val_result_group.setLayout(val_layout)
        # detect/segment/pose 通用标签
        self.val_map50_label = QLabel("mAP50: -")
        val_layout.addWidget(self.val_map50_label)
        self.val_map50_95_label = QLabel("mAP50-95: -")
        val_layout.addWidget(self.val_map50_95_label)
        # segment 任务额外显示 box 指标
        self.val_box_map50_label = QLabel("Box mAP50: -")
        self.val_box_map50_label.setVisible(False)
        val_layout.addWidget(self.val_box_map50_label)
        self.val_box_map50_95_label = QLabel("Box mAP50-95: -")
        self.val_box_map50_95_label.setVisible(False)
        val_layout.addWidget(self.val_box_map50_95_label)
        # classify 任务显示准确率
        self.val_top1_label = QLabel("Top-1 Acc: -")
        self.val_top1_label.setVisible(False)
        val_layout.addWidget(self.val_top1_label)
        self.val_top5_label = QLabel("Top-5 Acc: -")
        self.val_top5_label.setVisible(False)
        val_layout.addWidget(self.val_top5_label)
        # 任务类型标签
        self.val_task_label = QLabel("任务类型: -")
        val_layout.addWidget(self.val_task_label)
        self.val_result_group.setVisible(False)
        results_layout.addWidget(self.val_result_group)

        # --- Left panel: controls ---
        left_layout.addWidget(model_group)
        left_layout.addWidget(input_group)
        left_layout.addWidget(realsense_group)
        left_layout.addStretch()

        # --- Right panel: results ---
        right_layout.addWidget(self.result_label, 1)
        right_layout.addWidget(results_group)

        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([360, 800])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer_layout.addWidget(splitter)

    def _toggle_advanced_params(self) -> None:
        visible = self._advanced_toggle_btn.isChecked()
        for w in self._advanced_widgets:
            w.setVisible(visible)
        self._advanced_toggle_btn.setText("高级参数 ▲" if visible else "高级参数 ▼")

    def _get_advanced_params(self) -> dict:
        """获取高级预测参数"""
        return {
            "imgsz": self.imgsz_spin.value(),
            "device": self.device_combo.currentData() or "",
            "max_det": self.max_det_spin.value(),
        }

    def set_project_config(self, config: Optional[ProjectConfig]) -> None:
        if config is None:
            # 自由空间模式:清空
            self._project_config = None
            self.model_path_edit.setText("")
            if hasattr(self, '_batch_images'):
                self._batch_images = []
            if hasattr(self, '_batch_index'):
                self._batch_index = 0
            if hasattr(self, 'prev_btn'):
                self.prev_btn.setVisible(False)
            if hasattr(self, 'next_btn'):
                self.next_btn.setVisible(False)
            return
        self._project_config = config
        # 清空模型路径(无论新工作区间是否有 best.pt)
        runs_dir = Path(config.project_path) / "runs"
        best_path = ""
        if runs_dir.exists():
            best_paths = list(runs_dir.rglob("best.pt"))
            if best_paths:
                best_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                best_path = str(best_paths[0])
        self.model_path_edit.setText(best_path)
        # 清空批量推理图片列表
        if hasattr(self, '_batch_images'):
            self._batch_images = []
        if hasattr(self, '_batch_index'):
            self._batch_index = 0
        if hasattr(self, 'prev_btn'):
            self.prev_btn.setVisible(False)
        if hasattr(self, 'next_btn'):
            self.next_btn.setVisible(False)

    def _get_project_subdir(self, subdir: str) -> str:
        """返回工作区间下指定子目录路径,若不存在则回退到用户主目录。"""
        if self._project_config is not None:
            candidate = Path(self._project_config.project_path) / subdir
            if candidate.is_dir():
                return str(candidate)
        return str(Path.home())

    def _on_browse_model(self) -> None:
        start_dir = str(Path.home())
        if self._project_config is not None:
            models_dir = Path(self._project_config.project_path) / "models"
            runs_dir = Path(self._project_config.project_path) / "runs"
            if models_dir.is_dir():
                start_dir = str(models_dir)
            elif runs_dir.is_dir():
                start_dir = str(runs_dir)
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", start_dir, "模型文件 (*.pt *.onnx *.torchscript *.xml *.engine);;PyTorch (*.pt);;ONNX (*.onnx);;TorchScript (*.torchscript);;OpenVINO (*.xml);;TensorRT (*.engine);;所有文件 (*)")
        if path:
            self.model_path_edit.setText(path)

    _STATE_TEXTS = {
        "idle": "空闲",
        "loading": "加载中",
        "predicting": "推理中",
        "validating": "验证中",
        "exporting": "导出中",
    }

    def _set_model_state(self, state: str) -> None:
        """更新模型任务状态并刷新 UI 标签。"""
        self._model_state = state
        text = self._STATE_TEXTS.get(state, state)
        self.model_state_label.setText(f"状态: {text}")

    def _check_model_busy(self) -> bool:
        """检查是否有模型任务正在运行，弹窗提示并返回 True。"""
        state_labels = {
            self.STATE_LOADING: "模型加载中",
            self.STATE_PREDICTING: "实时推理中，请先点击停止",
            self.STATE_VALIDATING: "模型验证中",
            self.STATE_EXPORTING: "模型导出中",
        }
        label = state_labels.get(self._model_state)
        if label:
            QMessageBox.warning(self, "提示", f"{label}，请等待完成后再操作")
            return True
        return False

    def _on_load_model(self) -> None:
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            QMessageBox.warning(self, "验证失败", "请先输入或选择模型路径")
            return
        if self._check_model_busy():
            return
        # For ONNX files, try to infer or ask for task type (UI thread, 快速操作)
        task = ""
        if model_path.lower().endswith(".onnx"):
            task = self.predictor._guess_onnx_task(model_path)
            if not task:
                items = ["detect", "segment", "pose", "obb", "classify"]
                item, ok = QInputDialog.getItem(
                    self, "选择任务类型",
                    "无法自动推断 ONNX 模型的任务类型，请选择：\n\n"
                    "• detect — 目标检测\n"
                    "• segment — 实例分割\n"
                    "• pose — 关键点检测\n"
                    "• obb — 旋转框检测\n"
                    "• classify — 图像分类",
                    items, 0, False,
                )
                if ok:
                    task = item
                else:
                    task = "detect"
        # 后台线程加载模型，避免 ONNX warmup / CUDA 初始化阻塞 UI
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setText("加载中...")
        self._set_model_state(self.STATE_LOADING)
        self._model_load_worker = _ModelLoadWorker(self.predictor, model_path, task)
        self._model_load_worker.finished_signal.connect(self._on_model_loaded)
        self._model_load_worker.start()

    def _on_model_loaded(self, success: bool, model_path: str, info: dict) -> None:
        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setText("加载模型")
        self._model_load_worker = None
        self._set_model_state(self.STATE_IDLE)
        if success:
            task = info.get("task", "unknown")
            names = info.get("class_names", [])
            ext = Path(model_path).suffix.lower()
            format_names = {
                ".pt": "PyTorch", ".onnx": "ONNX",
                ".torchscript": "TorchScript", ".xml": "OpenVINO",
                ".engine": "TensorRT",
            }
            model_format = format_names.get(ext, ext)
            msg = f"模型加载成功\n格式: {model_format}\n任务类型: {task}\n类别数: {len(names)}"
            if names:
                msg += f"\n类别: {', '.join(names[:10])}"
                if len(names) > 10:
                    msg += "..."
            self.model_loaded.emit(self.predictor.model)
            QMessageBox.information(self, "成功", msg)
        else:
            QMessageBox.critical(self, "错误", "模型加载失败，请检查文件路径和格式")

    def _on_select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", self._get_project_subdir("images"),
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if not path:
            return
        self._batch_images = []
        self._batch_index = 0
        self.prev_btn.setVisible(False)
        self.next_btn.setVisible(False)
        self._run_image_predict(path)

    def _select_image_directory(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录", self._get_project_subdir("images"))
        if not dir_path:
            return
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self._batch_images = []
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    self._batch_images.append(os.path.join(root, f))
        self._batch_images.sort()
        if not self._batch_images:
            QMessageBox.warning(self, "提示", "所选目录中没有找到图片文件")
            return
        self._batch_index = 0
        self.prev_btn.setVisible(True)
        self.next_btn.setVisible(True)
        self._show_batch_image()

    def _show_batch_image(self) -> None:
        if not self._batch_images:
            return
        path = self._batch_images[self._batch_index]
        self._run_image_predict(path)

    def _prev_batch_image(self) -> None:
        if self._batch_index > 0:
            self._batch_index -= 1
            self._show_batch_image()

    def _next_batch_image(self) -> None:
        if self._batch_index < len(self._batch_images) - 1:
            self._batch_index += 1
            self._show_batch_image()

    def _run_image_predict(self, path: str) -> None:
        if self._check_model_busy():
            return
        self.result_label.setText("推理中...")
        self.image_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self._set_model_state(self.STATE_PREDICTING)
        adv_params = self._get_advanced_params()
        self._image_predict_worker = _ImagePredictWorker(
            self.predictor, path, self.conf_spin.value(), self.iou_spin.value(),
            imgsz=adv_params["imgsz"], device=adv_params["device"], max_det=adv_params["max_det"]
        )
        self._image_predict_worker.done_signal.connect(self._on_image_predict_done)
        self._image_predict_worker.error_signal.connect(self._on_image_predict_error)
        self._image_predict_worker.start()
        self._image_predict_worker.finished.connect(self._image_predict_worker.deleteLater)
        self._image_predict_worker.finished.connect(lambda: setattr(self, '_image_predict_worker', None))

    def _on_image_predict_done(self, annotated: np.ndarray, results: object) -> None:
        self.image_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self._set_model_state(self.STATE_IDLE)
        if annotated is not None and annotated.size > 0:
            self._display_np_image(annotated)
            count = 0
            if results is not None:
                try:
                    count = len(results.boxes)
                except Exception:
                    count = 0
            self.det_count_label.setText(f"检测数量: {count}")
            if self._batch_images:
                self.fps_label.setText(f"图片: {self._batch_index + 1}/{len(self._batch_images)}")
            else:
                self.fps_label.setText("FPS: -")
        else:
            QMessageBox.warning(self, "警告", "图片读取或推理失败")

    def _on_image_predict_error(self, msg: str) -> None:
        self.image_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.next_btn.setEnabled(True)
        self._set_model_state(self.STATE_IDLE)
        QMessageBox.warning(self, "警告", f"图片推理出错:\n{msg}")

    def _on_select_video(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", self._get_project_subdir("images"),
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv)"
        )
        if not path:
            return
        self._start_capture(path)

    def _on_open_camera(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        self._start_capture(0)

    def _start_capture(self, source: Union[str, int]) -> None:
        if self._check_model_busy():
            return
        self._on_stop()
        self._batch_images = []
        self._batch_index = 0
        self.prev_btn.setVisible(False)
        self.next_btn.setVisible(False)
        if not self._inference_worker.isRunning():
            self._inference_worker.start()
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "无法打开视频源")
            self.cap = None
            return
        self._set_model_state(self.STATE_PREDICTING)
        self.stop_btn.setEnabled(True)
        self.image_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.camera_btn.setEnabled(False)
        self._last_frame_time = time.time()
        self.timer.start(30)

    def _on_refresh_rs_devices(self) -> None:
        self.rs_device_combo.clear()
        if not RealSenseCamera.is_available():
            self.rs_device_combo.addItem("未安装 pyrealsense2")
            self.rs_camera_btn.setEnabled(False)
            QMessageBox.warning(self, "提示", "未检测到 pyrealsense2 库，请先安装:\npip install pyrealsense2")
            return
        devices = RealSenseCamera.list_devices()
        if not devices:
            self.rs_device_combo.addItem("未检测到设备")
            self.rs_camera_btn.setEnabled(False)
            return
        for dev in devices:
            self.rs_device_combo.addItem(f"{dev.name} ({dev.serial})", dev.serial)
        self.rs_camera_btn.setEnabled(True)

    def _on_open_rs_camera(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if self._check_model_busy():
            return
        self._on_stop()
        self._batch_images = []
        self._batch_index = 0
        self.prev_btn.setVisible(False)
        self.next_btn.setVisible(False)
        if not self._inference_worker.isRunning():
            self._inference_worker.start()
        serial = self.rs_device_combo.currentData()
        if serial is None:
            QMessageBox.warning(self, "警告", "请先选择一个 RealSense 设备")
            return
        success = self.rs_camera.start(device_serial=serial)
        if not success:
            QMessageBox.warning(self, "警告", "无法打开 RealSense 设备")
            return
        self._set_model_state(self.STATE_PREDICTING)
        self.stop_btn.setEnabled(True)
        self.image_btn.setEnabled(False)
        self.video_btn.setEnabled(False)
        self.camera_btn.setEnabled(False)
        self.rs_camera_btn.setEnabled(False)
        self.rs_device_combo.setEnabled(False)
        self.rs_refresh_btn.setEnabled(False)
        self._last_frame_time = time.time()
        self.timer.start(30)

    def _on_depth_check_toggled(self, checked: bool) -> None:
        self._show_depth = checked

    def _on_inference_result(self, annotated: np.ndarray, results: object) -> None:
        if annotated is not None and annotated.size > 0:
            self._display_np_image(annotated)
        elif self._last_frame is not None and self._last_frame.size > 0:
            self._display_np_image(self._last_frame)
        count = 0
        try:
            if results is not None:
                count = len(results.boxes)
        except Exception:
            count = 0
        self.det_count_label.setText(f"检测数量: {count}")
        self.fps_label.setText(f"FPS: {self._fps:.1f}")

    def _on_timer_timeout(self) -> None:
        frame = None
        depth_np = None
        if self.rs_camera.running:
            color_np, depth_np = self.rs_camera.get_frames()
            if color_np is None:
                self._on_stop()
                return
            frame = color_np
            self._last_frame = frame
        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self._on_stop()
                return
            self._last_frame = frame
        else:
            self._on_stop()
            return

        current_time = time.time()
        delta = current_time - self._last_frame_time
        if delta > 0:
            self._fps = 1.0 / delta
        self._last_frame_time = current_time

        if self._show_depth and depth_np is not None:
            colorized = self.rs_camera.colorize_depth(depth_np)
            if colorized is not None:
                self._display_np_image(colorized)
            else:
                if not self._inference_worker.is_busy:
                    adv_params = self._get_advanced_params()
                    self._inference_worker.submit(
                        frame, self.conf_spin.value(), self.iou_spin.value(),
                        imgsz=adv_params["imgsz"], device=adv_params["device"], max_det=adv_params["max_det"]
                    )
        else:
            if self._inference_worker.is_busy:
                self.fps_label.setText(f"FPS: {self._fps:.1f}")
            else:
                adv_params = self._get_advanced_params()
                self._inference_worker.submit(
                    frame, self.conf_spin.value(), self.iou_spin.value(),
                    imgsz=adv_params["imgsz"], device=adv_params["device"], max_det=adv_params["max_det"]
                )

    def _on_stop(self) -> None:
        self.timer.stop()
        if self._inference_worker.isRunning():
            self._inference_worker.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.rs_camera.running:
            self.rs_camera.stop()
        # 仅当处于实时推理状态时才重置，避免覆盖 validate/export 状态
        if self._model_state == self.STATE_PREDICTING:
            self._set_model_state(self.STATE_IDLE)
        self.stop_btn.setEnabled(False)
        self.image_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.camera_btn.setEnabled(True)
        self.rs_camera_btn.setEnabled(True)
        self.rs_device_combo.setEnabled(True)
        self.rs_refresh_btn.setEnabled(True)
        self.depth_check.setChecked(False)
        self._show_depth = False

    def closeEvent(self, event: QCloseEvent) -> None:
        self._on_stop()
        # 先请求所有工作线程协作停止，再同步等待退出
        for worker in (self._validate_worker, self._export_worker, self._image_predict_worker):
            if worker is not None:
                worker.request_stop()
        # closeEvent 中同步等待线程退出，防止析构时崩溃
        if self._inference_worker is not None and self._inference_worker.isRunning():
            self._inference_worker.wait(5000)
        if self._model_load_worker is not None and self._model_load_worker.isRunning():
            self._model_load_worker.wait(5000)
        if self._validate_worker is not None and self._validate_worker.isRunning():
            self._validate_worker.wait(5000)
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.wait(5000)
        if self._image_predict_worker is not None and self._image_predict_worker.isRunning():
            self._image_predict_worker.wait(5000)
        super().closeEvent(event)

    def _on_validate(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if self._check_model_busy():
            return
        data_path = ""
        if self._project_config is not None:
            data_yaml = Path(self._project_config.project_path) / "datasets" / "data.yaml"
            if data_yaml.exists():
                data_path = str(data_yaml)
        if not data_path:
            data_path, _ = QFileDialog.getOpenFileName(self, "选择数据集配置文件", self._get_project_subdir("datasets"), "YAML (*.yaml *.yml)")
        if not data_path:
            return
        self.validate_btn.setEnabled(False)
        self.validate_btn.setText("验证中...")
        self._set_model_state(self.STATE_VALIDATING)
        self._validate_worker = _ValidateWorker(self.predictor, data_path)
        self._validate_worker.done_signal.connect(self._on_validate_done)
        self._validate_worker.error_signal.connect(self._on_validate_error)
        self._validate_worker.finished.connect(lambda: setattr(self, '_validate_worker', None))
        self._validate_worker.start()
        self._validate_worker.finished.connect(self._validate_worker.deleteLater)

    def _on_validate_done(self, metrics: dict) -> None:
        self.validate_btn.setEnabled(True)
        self.validate_btn.setText("验证模型")
        self._set_model_state(self.STATE_IDLE)
        if metrics:
            task = metrics.get("task", "detect")
            self.val_task_label.setText(f"任务类型: {task}")
            # 先隐藏所有特定任务标签
            self.val_map50_label.setVisible(False)
            self.val_map50_95_label.setVisible(False)
            self.val_box_map50_label.setVisible(False)
            self.val_box_map50_95_label.setVisible(False)
            self.val_top1_label.setVisible(False)
            self.val_top5_label.setVisible(False)
            # 根据任务类型显示对应指标
            if task == "detect":
                self.val_map50_label.setText(f"mAP50 (box): {metrics.get('map50', 0.0):.4f}")
                self.val_map50_95_label.setText(f"mAP50-95 (box): {metrics.get('map50_95', 0.0):.4f}")
                self.val_map50_label.setVisible(True)
                self.val_map50_95_label.setVisible(True)
            elif task == "segment":
                self.val_map50_label.setText(f"mAP50 (mask): {metrics.get('map50', 0.0):.4f}")
                self.val_map50_95_label.setText(f"mAP50-95 (mask): {metrics.get('map50_95', 0.0):.4f}")
                self.val_map50_label.setVisible(True)
                self.val_map50_95_label.setVisible(True)
                # 可选显示 box 指标
                if "box_map50" in metrics:
                    self.val_box_map50_label.setText(f"mAP50 (box): {metrics.get('box_map50', 0.0):.4f}")
                    self.val_box_map50_label.setVisible(True)
                if "box_map50_95" in metrics:
                    self.val_box_map50_95_label.setText(f"mAP50-95 (box): {metrics.get('box_map50_95', 0.0):.4f}")
                    self.val_box_map50_95_label.setVisible(True)
            elif task == "pose":
                self.val_map50_label.setText(f"mAP50 (pose): {metrics.get('map50', 0.0):.4f}")
                self.val_map50_95_label.setText(f"mAP50-95 (pose): {metrics.get('map50_95', 0.0):.4f}")
                self.val_map50_label.setVisible(True)
                self.val_map50_95_label.setVisible(True)
            elif task == "classify":
                self.val_top1_label.setText(f"Top-1 Acc: {metrics.get('top1', 0.0):.4f}")
                self.val_top5_label.setText(f"Top-5 Acc: {metrics.get('top5', 0.0):.4f}")
                self.val_top1_label.setVisible(True)
                self.val_top5_label.setVisible(True)
            else:
                # 其他任务（如 obb）使用 box 指标
                self.val_map50_label.setText(f"mAP50: {metrics.get('map50', 0.0):.4f}")
                self.val_map50_95_label.setText(f"mAP50-95: {metrics.get('map50_95', 0.0):.4f}")
                self.val_map50_label.setVisible(True)
                self.val_map50_95_label.setVisible(True)
            self.val_result_group.setVisible(True)
        else:
            QMessageBox.warning(self, "警告", "模型验证失败，请检查数据集配置")

    def _on_validate_error(self, msg: str) -> None:
        self.validate_btn.setEnabled(True)
        self.validate_btn.setText("验证模型")
        self._set_model_state(self.STATE_IDLE)
        QMessageBox.warning(self, "警告", f"模型验证出错:\n{msg}")

    def _on_export_clicked(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        if self._check_model_busy():
            return
        task = self.predictor.get_model_info().get("task", "")
        dlg = ExportDialog(task=task, parent=self)
        dlg.export_requested.connect(self._on_dialog_export)
        self._export_dialog = dlg
        dlg.exec()

    def _on_dialog_export(self, fmt: str, kwargs: dict) -> None:
        output_dir = ""
        if self._project_config is not None:
            output_dir = str(Path(self._project_config.project_path) / "models")
        if not output_dir:
            output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return
        self._set_model_state(self.STATE_EXPORTING)
        self._export_worker = _ExportWorker(self.predictor, fmt, output_dir, **kwargs)
        self._export_worker.done_signal.connect(self._on_export_done)
        self._export_worker.error_signal.connect(self._on_export_error)
        self._export_worker.start()
        self._export_worker.finished.connect(self._export_worker.deleteLater)
        self._export_worker.finished.connect(lambda: setattr(self, '_export_worker', None))

    def _on_export_done(self, exported_path: str, success: bool, error_msg: str) -> None:
        self._set_model_state(self.STATE_IDLE)
        dlg = getattr(self, '_export_dialog', None)
        if dlg is not None:
            dlg.accept()
        if exported_path:
            task = self.predictor.get_model_info().get("task", "")
            if success:
                reply = QMessageBox.question(
                    self, "导出成功",
                    f"模型已导出至:\n{exported_path}\n\n是否加载导出模型进行测试？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.model_path_edit.setText(exported_path)
                    success_load = self.predictor.load_model(exported_path, task=task)
                    if success_load:
                        info = self.predictor.get_model_info()
                        model_task = info.get("task", "unknown")
                        names = info.get("class_names", [])
                        format_name = self._exported_format_name(exported_path)
                        msg = f"模型加载成功\n格式: {format_name}\n任务类型: {model_task}\n类别数: {len(names)}"
                        if names:
                            msg += f"\n类别: {', '.join(names[:10])}"
                            if len(names) > 10:
                                msg += "..."
                        self.model_loaded.emit(self.predictor.model)
                        QMessageBox.information(self, "成功", msg)
                    else:
                        QMessageBox.critical(self, "错误", "模型加载失败，请检查文件路径和格式")
            else:
                # Export succeeded but verification failed
                reply = QMessageBox.question(
                    self, "导出成功但验证失败",
                    f"模型已导出至:\n{exported_path}\n\n验证失败: {error_msg}\n\n"
                    f"导出的模型文件可能无法正常推理，是否仍要加载测试？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.model_path_edit.setText(exported_path)
                    success_load = self.predictor.load_model(exported_path, task=task)
                    if success_load:
                        info = self.predictor.get_model_info()
                        model_task = info.get("task", "unknown")
                        names = info.get("class_names", [])
                        format_name = self._exported_format_name(exported_path)
                        msg = f"模型加载成功\n格式: {format_name}\n任务类型: {model_task}\n类别数: {len(names)}"
                        if names:
                            msg += f"\n类别: {', '.join(names[:10])}"
                            if len(names) > 10:
                                msg += "..."
                        self.model_loaded.emit(self.predictor.model)
                        QMessageBox.information(self, "成功", msg)
                    else:
                        QMessageBox.critical(self, "错误", "模型加载失败，请检查文件路径和格式")
        else:
            QMessageBox.critical(self, "错误", "模型导出失败")

    def _on_export_error(self, msg: str) -> None:
        self._set_model_state(self.STATE_IDLE)
        dlg = getattr(self, '_export_dialog', None)
        if dlg is not None:
            dlg._confirm_btn.setEnabled(True)
            dlg._confirm_btn.setText("确认导出")
        QMessageBox.critical(self, "错误", f"模型导出出错:\n{msg}")

    @staticmethod
    def _exported_format_name(path: str) -> str:
        return {
            ".onnx": "ONNX",
            ".engine": "TensorRT",
            ".torchscript": "TorchScript",
            ".xml": "OpenVINO",
        }.get(Path(path).suffix.lower(), Path(path).suffix.lstrip(".").upper())

    def _display_np_image(self, image_np: np.ndarray) -> None:
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        elif image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = np.ascontiguousarray(image_np)
        h, w, ch = image_np.shape
        bytes_per_line = ch * w
        q_img = QImage(image_np.copy(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.result_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.result_label.setPixmap(scaled)
