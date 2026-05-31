import os
import time
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QObject, QThread, QWaitCondition, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yolo26_app.core.config import ProjectConfig
from yolo26_app.core.predictor import YOLOPredictor
from yolo26_app.core.realsense_camera import RealSenseCamera


class _ValidateWorker(QThread):
    done_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, predictor: YOLOPredictor, data_path: str) -> None:
        super().__init__()
        self.predictor = predictor
        self.data_path = data_path

    def run(self) -> None:
        try:
            result = self.predictor.validate_model(self.data_path)
            self.done_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


class _ExportWorker(QThread):
    done_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, predictor: YOLOPredictor, format: str, output_dir: str) -> None:
        super().__init__()
        self.predictor = predictor
        self.format = format
        self.output_dir = output_dir

    def run(self) -> None:
        try:
            exported_path = self.predictor.export_model(self.format, self.output_dir)
            self.done_signal.emit(exported_path)
        except Exception as e:
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
        self._busy: bool = False
        self._stop_flag: bool = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def submit(self, frame: np.ndarray, conf: float) -> None:
        self._mutex.lock()
        self._frame = frame.copy()
        self._conf = conf
        self._busy = True
        self._cond.wakeOne()
        self._mutex.unlock()

    def stop(self) -> None:
        self._mutex.lock()
        self._stop_flag = True
        self._cond.wakeOne()
        self._mutex.unlock()
        self.wait()
        self._mutex.lock()
        self._stop_flag = False
        self._frame = None
        self._busy = False
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
            self._frame = None
            self._mutex.unlock()

            try:
                annotated, results = self._predictor.predict_frame(frame, conf=conf)
            except Exception:
                annotated = frame
                results = None
            self.result_signal.emit(annotated, results)

            self._mutex.lock()
            self._busy = False
            self._mutex.unlock()


class TestWidget(QWidget):
    model_loaded = pyqtSignal(object)

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
        self._inference_worker = _InferenceWorker(self.predictor)
        self._inference_worker.result_signal.connect(self._on_inference_result)
        self._last_frame: Optional[np.ndarray] = None
        self._init_ui()

    def __del__(self) -> None:
        if self._inference_worker.isRunning():
            self._inference_worker.stop()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        model_group = QGroupBox("模型设置")
        model_form = QFormLayout()
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

        self.load_model_btn = QPushButton("加载模型")
        self.load_model_btn.clicked.connect(self._on_load_model)
        model_form.addRow(self.load_model_btn)

        main_layout.addWidget(model_group)

        input_group = QGroupBox("推理输入")
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

        rs_layout = QHBoxLayout()
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
        input_layout.addLayout(rs_layout)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        input_layout.addWidget(self.stop_btn)

        main_layout.addWidget(input_group)

        self.result_label = QLabel("等待推理...")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(640, 480)
        self.result_label.setStyleSheet("QLabel { background-color: #1a1a2e; color: #aaa; }")
        main_layout.addWidget(self.result_label, stretch=1)

        results_group = QGroupBox("结果与指标")
        results_layout = QVBoxLayout()
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
        self.val_map50_label = QLabel("mAP50: -")
        val_layout.addWidget(self.val_map50_label)
        self.val_map50_95_label = QLabel("mAP50-95: -")
        val_layout.addWidget(self.val_map50_95_label)
        self.val_result_group.setVisible(False)
        results_layout.addWidget(self.val_result_group)

        self.export_settings_group = QGroupBox("导出设置")
        export_settings_layout = QVBoxLayout()
        self.export_settings_group.setLayout(export_settings_layout)
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("格式:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["onnx", "torchscript", "openvino", "engine"])
        format_layout.addWidget(self.export_format_combo)
        export_settings_layout.addLayout(format_layout)
        self.confirm_export_btn = QPushButton("确认导出")
        self.confirm_export_btn.clicked.connect(self._on_confirm_export)
        export_settings_layout.addWidget(self.confirm_export_btn)
        self.export_settings_group.setVisible(False)
        results_layout.addWidget(self.export_settings_group)

        main_layout.addWidget(results_group)

    def set_project_config(self, config: ProjectConfig) -> None:
        self._project_config = config
        runs_dir = Path(config.project_path) / "runs"
        if runs_dir.exists():
            best_paths = list(runs_dir.rglob("best.pt"))
            if best_paths:
                best_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                self.model_path_edit.setText(str(best_paths[0]))

    def _on_browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt *.onnx *.torchscript *.xml *.engine);;PyTorch (*.pt);;ONNX (*.onnx);;TorchScript (*.torchscript);;OpenVINO (*.xml);;TensorRT (*.engine);;所有文件 (*)")
        if path:
            self.model_path_edit.setText(path)

    def _on_load_model(self) -> None:
        path = self.model_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "警告", "请先选择模型文件路径")
            return
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setText("加载中...")
        success = self.predictor.load_model(path)
        if success:
            info = self.predictor.get_model_info()
            task = info.get("task", "unknown")
            names = info.get("class_names", [])
            ext = Path(path).suffix.lower()
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
        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setText("加载模型")

    def _on_select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp)"
        )
        if not path:
            return
        self._batch_images = []
        self._batch_index = 0
        self.prev_btn.setVisible(False)
        self.next_btn.setVisible(False)
        conf = self.conf_spin.value()
        annotated, results = self.predictor.predict_image(path, conf=conf)
        if annotated is not None and annotated.size > 0:
            self._display_np_image(annotated)
            count = 0
            if results is not None:
                try:
                    count = len(results.boxes)
                except Exception:
                    count = 0
            self.det_count_label.setText(f"检测数量: {count}")
            self.fps_label.setText("FPS: -")
        else:
            QMessageBox.warning(self, "警告", "图片读取或推理失败")

    def _select_image_directory(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
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
        annotated, results = self.predictor.predict_image(path, conf=self.conf_spin.value())
        if annotated is not None and annotated.size > 0:
            self._display_np_image(annotated)
        det_count = 0
        if results is not None:
            try:
                det_count = len(results.boxes)
            except Exception:
                det_count = 0
        self.det_count_label.setText(f"检测数量: {det_count}")
        self.fps_label.setText(f"图片: {self._batch_index + 1}/{len(self._batch_images)}")

    def _prev_batch_image(self) -> None:
        if self._batch_index > 0:
            self._batch_index -= 1
            self._show_batch_image()

    def _next_batch_image(self) -> None:
        if self._batch_index < len(self._batch_images) - 1:
            self._batch_index += 1
            self._show_batch_image()

    def _on_select_video(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
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
                    self._inference_worker.submit(frame, self.conf_spin.value())
        else:
            if self._inference_worker.is_busy:
                self.fps_label.setText(f"FPS: {self._fps:.1f}")
            else:
                self._inference_worker.submit(frame, self.conf_spin.value())

    def _on_stop(self) -> None:
        self.timer.stop()
        if self._inference_worker.isRunning():
            self._inference_worker.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.rs_camera.running:
            self.rs_camera.stop()
        self.stop_btn.setEnabled(False)
        self.image_btn.setEnabled(True)
        self.video_btn.setEnabled(True)
        self.camera_btn.setEnabled(True)
        self.rs_camera_btn.setEnabled(True)
        self.rs_device_combo.setEnabled(True)
        self.rs_refresh_btn.setEnabled(True)
        self.depth_check.setChecked(False)
        self._show_depth = False

    def _on_validate(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        data_path = ""
        if self._project_config is not None:
            data_yaml = Path(self._project_config.project_path) / "datasets" / "data.yaml"
            if data_yaml.exists():
                data_path = str(data_yaml)
        if not data_path:
            data_path, _ = QFileDialog.getOpenFileName(self, "选择数据集配置文件", "", "YAML (*.yaml *.yml)")
        if not data_path:
            return
        self.validate_btn.setEnabled(False)
        self.validate_btn.setText("验证中...")
        self._validate_worker = _ValidateWorker(self.predictor, data_path)
        self._validate_worker.done_signal.connect(self._on_validate_done)
        self._validate_worker.error_signal.connect(self._on_validate_error)
        self._validate_worker.start()

    def _on_validate_done(self, metrics: dict) -> None:
        self.validate_btn.setEnabled(True)
        self.validate_btn.setText("验证模型")
        if metrics:
            self.val_map50_label.setText(f"mAP50: {metrics.get('map50', 0.0):.4f}")
            self.val_map50_95_label.setText(f"mAP50-95: {metrics.get('map50_95', 0.0):.4f}")
            self.val_result_group.setVisible(True)
        else:
            QMessageBox.warning(self, "警告", "模型验证失败，请检查数据集配置")

    def _on_validate_error(self, msg: str) -> None:
        self.validate_btn.setEnabled(True)
        self.validate_btn.setText("验证模型")
        QMessageBox.warning(self, "警告", f"模型验证出错:\n{msg}")

    def _on_export_clicked(self) -> None:
        if self.predictor.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        self.export_settings_group.setVisible(True)

    def _on_confirm_export(self) -> None:
        fmt = self.export_format_combo.currentText()
        output_dir = ""
        if self._project_config is not None:
            output_dir = str(Path(self._project_config.project_path) / "models")
        if not output_dir:
            output_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not output_dir:
            return
        self.confirm_export_btn.setEnabled(False)
        self.confirm_export_btn.setText("导出中...")
        self._export_worker = _ExportWorker(self.predictor, fmt, output_dir)
        self._export_worker.done_signal.connect(self._on_export_done)
        self._export_worker.error_signal.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_done(self, exported_path: str) -> None:
        self.confirm_export_btn.setEnabled(True)
        self.confirm_export_btn.setText("确认导出")
        if exported_path:
            QMessageBox.information(self, "成功", f"模型已导出至:\n{exported_path}")
            self.export_settings_group.setVisible(False)
        else:
            QMessageBox.critical(self, "错误", "模型导出失败")

    def _on_export_error(self, msg: str) -> None:
        self.confirm_export_btn.setEnabled(True)
        self.confirm_export_btn.setText("确认导出")
        QMessageBox.critical(self, "错误", f"模型导出出错:\n{msg}")

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
