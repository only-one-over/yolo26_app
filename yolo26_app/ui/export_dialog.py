from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

FORMAT_PARAMS = {
    "onnx": {"imgsz", "half", "dynamic", "batch", "opset", "simplify", "nms"},
    "torchscript": {"imgsz", "half", "dynamic", "batch"},
    "openvino": {"imgsz", "half", "int8", "data", "dynamic", "batch"},
    "engine": {
        "imgsz", "precision", "data", "fraction", "device",
        "dynamic", "batch", "workspace", "simplify", "nms",
    },
    "coreml": {"imgsz", "half", "int8", "data", "dynamic", "batch"},
    "tflite": {"imgsz", "half", "int8", "data", "batch"},
    "ncnn": {"imgsz", "half", "batch"},
    "paddle": {"imgsz", "batch"},
    "mnn": {"imgsz", "half", "int8", "data", "batch"},
    "rknn": {"imgsz", "batch"},
}

FORMAT_DISPLAY_NAMES = {
    "onnx": "ONNX (.onnx)",
    "torchscript": "TorchScript",
    "openvino": "OpenVINO",
    "engine": "TensorRT (.engine)",
    "coreml": "CoreML",
    "tflite": "TensorFlow Lite",
    "ncnn": "NCNN",
    "paddle": "PaddlePaddle",
    "mnn": "MNN",
    "rknn": "RKNN",
}

EXPORT_PRESETS = {
    "自定义": None,
    "TensorRT FP16": {
        "format": "engine", "precision": 16, "imgsz": 640,
        "dynamic": True, "batch": 1, "workspace": 0,
        "simplify": True, "nms": False, "device": "0",
    },
    "TensorRT INT8": {
        "format": "engine", "precision": 8, "imgsz": 640,
        "dynamic": True, "batch": 8, "workspace": 4,
        "simplify": True, "nms": False, "device": "0",
    },
    "检测 (Detect)": {"format": "onnx", "imgsz": 640, "half": True, "dynamic": False, "simplify": True, "opset": 17, "nms": False, "batch": 1},
    "实例分割 (Segment)": {"format": "onnx", "imgsz": 640, "half": False, "dynamic": False, "simplify": False, "opset": 17, "nms": False},
    "关键点 (Pose)": {"format": "onnx", "imgsz": 640, "half": True, "dynamic": False, "simplify": True, "opset": 17},
    "旋转框 (OBB)": {"format": "onnx", "imgsz": 1024, "half": True, "simplify": True, "opset": 17},
    "分类 (Classify)": {"format": "onnx", "imgsz": 224, "half": True, "simplify": True},
}

TASK_PRESET_MAP = {
    "detect": "检测 (Detect)",
    "segment": "实例分割 (Segment)",
    "pose": "关键点 (Pose)",
    "obb": "旋转框 (OBB)",
    "classify": "分类 (Classify)",
}

TASK_DISPLAY_NAMES = {
    "detect": "检测",
    "segment": "实例分割",
    "pose": "关键点检测",
    "obb": "旋转框检测",
    "classify": "分类",
}


class ExportDialog(QDialog):
    export_requested = pyqtSignal(str, dict)

    def __init__(self, task="", parent=None):
        super().__init__(parent)
        self._model_task = task
        self.setWindowTitle("导出模型")
        self.setModal(True)
        self.setMinimumSize(560, 640)

        self._param_widgets = {}

        layout = QVBoxLayout(self)

        # --- 任务预设 ---
        preset_group = QGroupBox("任务预设")
        preset_group.setObjectName("configCard")
        preset_layout = QVBoxLayout(preset_group)
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(EXPORT_PRESETS.keys()))
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self._preset_combo)

        # 任务标签
        self._task_label = QLabel()
        if task:
            task_display = TASK_DISPLAY_NAMES.get(task, task)
            self._task_label.setText(f"当前模型任务: {task_display}")
        preset_layout.addWidget(self._task_label)

        # 不匹配警告标签
        self._warning_label = QLabel()
        self._warning_label.setObjectName("warningLabel")
        self._warning_label.setVisible(False)
        preset_layout.addWidget(self._warning_label)

        layout.addWidget(preset_group)

        # --- 导出格式 ---
        format_group = QGroupBox("导出格式")
        format_group.setObjectName("configCard")
        format_layout = QHBoxLayout(format_group)
        self._format_combo = QComboBox()
        for format_key in FORMAT_PARAMS:
            self._format_combo.addItem(FORMAT_DISPLAY_NAMES.get(format_key, format_key), format_key)
        self._format_combo.currentIndexChanged.connect(
            lambda: self._update_params_visibility(self._current_format())
        )
        format_layout.addWidget(self._format_combo)
        layout.addWidget(format_group)

        # --- 导出参数 ---
        params_group = QGroupBox("导出参数")
        params_group.setObjectName("configCard")
        params_layout = QVBoxLayout(params_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        form = QFormLayout(scroll_content)

        # imgsz
        self._imgsz_combo = QComboBox()
        self._imgsz_combo.addItems(["224", "320", "480", "640", "960", "1024", "1280"])
        self._imgsz_combo.setCurrentText("640")
        self._param_widgets["imgsz"] = self._imgsz_combo
        form.addRow("输入尺寸 (imgsz)", self._imgsz_combo)

        # half
        self._half_check = QCheckBox("FP16 半精度")
        self._param_widgets["half"] = self._half_check
        form.addRow("", self._half_check)

        # int8
        self._int8_check = QCheckBox("INT8 量化")
        self._int8_check.stateChanged.connect(self._on_int8_changed)
        self._param_widgets["int8"] = self._int8_check
        form.addRow("", self._int8_check)

        # TensorRT precision (current Ultralytics uses quantize=16/8).
        self._precision_combo = QComboBox()
        self._precision_combo.addItem("FP32", 32)
        self._precision_combo.addItem("FP16", 16)
        self._precision_combo.addItem("INT8", 8)
        self._precision_combo.setCurrentIndex(1)
        self._precision_combo.currentIndexChanged.connect(self._on_precision_changed)
        self._param_widgets["precision"] = self._precision_combo
        form.addRow("TensorRT 精度", self._precision_combo)

        # int8 data yaml
        self._data_layout = QHBoxLayout()
        self._data_edit = QLineEdit()
        self._data_edit.setPlaceholderText("选择校准数据集 YAML 文件")
        self._data_btn = QPushButton("浏览")
        self._data_btn.clicked.connect(self._on_browse_data)
        self._data_layout.addWidget(self._data_edit)
        self._data_layout.addWidget(self._data_btn)
        self._param_widgets["data"] = self._data_edit
        form.addRow("校准数据 (data)", self._data_layout)

        self._fraction_spin = QDoubleSpinBox()
        self._fraction_spin.setRange(0.05, 1.0)
        self._fraction_spin.setSingleStep(0.05)
        self._fraction_spin.setValue(1.0)
        self._fraction_spin.setDecimals(2)
        self._param_widgets["fraction"] = self._fraction_spin
        form.addRow("校准比例 (fraction)", self._fraction_spin)

        # int8 hint label
        self._int8_hint_label = QLabel("INT8 量化需要校准数据集")
        self._int8_hint_label.setObjectName("warningLabel")
        self._int8_hint_label.setVisible(False)
        form.addRow("", self._int8_hint_label)

        # dynamic
        self._dynamic_check = QCheckBox("动态输入尺寸")
        self._param_widgets["dynamic"] = self._dynamic_check
        form.addRow("", self._dynamic_check)

        # batch
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(1)
        self._param_widgets["batch"] = self._batch_spin
        form.addRow("Batch", self._batch_spin)

        self._device_edit = QLineEdit("0")
        self._device_edit.setPlaceholderText("例如 0、1、dla:0")
        self._param_widgets["device"] = self._device_edit
        form.addRow("导出设备 (device)", self._device_edit)

        # opset
        self._opset_spin = QSpinBox()
        self._opset_spin.setRange(9, 21)
        self._opset_spin.setValue(17)
        self._param_widgets["opset"] = self._opset_spin
        form.addRow("Opset", self._opset_spin)

        # workspace
        self._workspace_spin = QDoubleSpinBox()
        self._workspace_spin.setRange(0.0, 64.0)
        self._workspace_spin.setValue(0.0)
        self._workspace_spin.setSpecialValueText("自动")
        self._workspace_spin.setSuffix(" GiB")
        self._param_widgets["workspace"] = self._workspace_spin
        form.addRow("Workspace", self._workspace_spin)

        # simplify
        self._simplify_check = QCheckBox("图简化 (simplify)")
        self._simplify_check.setChecked(True)
        self._param_widgets["simplify"] = self._simplify_check
        form.addRow("", self._simplify_check)

        # nms
        self._nms_check = QCheckBox("NMS (非极大值抑制)")
        self._param_widgets["nms"] = self._nms_check
        form.addRow("", self._nms_check)

        self._engine_hint_label = QLabel(
            "TensorRT 导出需要 NVIDIA CUDA GPU；生成的 .engine 与导出设备及 TensorRT 版本相关。"
        )
        self._engine_hint_label.setWordWrap(True)
        self._engine_hint_label.setObjectName("warningLabel")
        self._engine_hint_label.setVisible(False)
        form.addRow("", self._engine_hint_label)

        scroll.setWidget(scroll_content)
        params_layout.addWidget(scroll)
        layout.addWidget(params_group)

        # --- 底部按钮 ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.setContentsMargins(12, 8, 12, 12)
        self._confirm_btn = QPushButton("确认导出")
        self._confirm_btn.setObjectName("primaryButton")
        self._confirm_btn.clicked.connect(self._on_confirm)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self._confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # 初始可见性
        self._update_params_visibility(self._current_format())

        # 根据模型任务自动选择预设
        if task in TASK_PRESET_MAP:
            self._preset_combo.setCurrentText(TASK_PRESET_MAP[task])

    # ------------------------------------------------------------------
    def _update_params_visibility(self, fmt: str):
        visible = FORMAT_PARAMS.get(fmt, set())
        for name, widget in self._param_widgets.items():
            if name == "data":
                data_visible = self._is_int8_selected(fmt)
                self._data_edit.setVisible(data_visible)
                self._data_btn.setVisible(data_visible)
                # 隐藏/显示 data 行的 label
                row = self._find_form_row("data")
                if row is not None:
                    label_item = self._form_layout().itemAt(row, QFormLayout.ItemRole.LabelRole)
                    if label_item and label_item.widget():
                        label_item.widget().setVisible(data_visible)
                # 更新 int8 提示
                self._update_int8_hint()
            elif name == "fraction":
                fraction_visible = name in visible and self._is_int8_selected(fmt)
                widget.setVisible(fraction_visible)
                row = self._find_form_row(name)
                if row is not None:
                    label_item = self._form_layout().itemAt(row, QFormLayout.ItemRole.LabelRole)
                    if label_item and label_item.widget():
                        label_item.widget().setVisible(fraction_visible)
            else:
                widget.setVisible(name in visible)
                # 同时隐藏对应的 label（QFormLayout 的 label item）
                row = self._find_form_row(name)
                if row is not None:
                    label_item = self._form_layout().itemAt(row, QFormLayout.ItemRole.LabelRole)
                    if label_item and label_item.widget():
                        label_item.widget().setVisible(name in visible)
        # int8 hint label 的可见性跟随 data 控件
        self._update_int8_hint()
        self._engine_hint_label.setVisible(fmt == "engine")
        engine_int8 = fmt == "engine" and self._precision_combo.currentData() == 8
        if engine_int8:
            self._dynamic_check.setChecked(True)
        self._dynamic_check.setEnabled(not engine_int8)
        self._dynamic_check.setToolTip(
            "TensorRT INT8 按官方要求启用动态输入尺寸" if engine_int8 else ""
        )

    def _find_form_row(self, param_name: str):
        form = self._form_layout()
        target = self._param_widgets.get(param_name)
        for row in range(form.rowCount()):
            field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if field_item is None:
                continue
            # 字段可能是 widget 或 layout
            if field_item.widget() is target:
                return row
            if field_item.layout() is not None:
                # 对于 data 参数，field 是包含 QLineEdit 的 QHBoxLayout
                layout = field_item.layout()
                for i in range(layout.count()):
                    if layout.itemAt(i).widget() is target:
                        return row
        return None

    def _form_layout(self) -> QFormLayout:
        scroll_content = self._param_widgets["imgsz"].parentWidget()
        return scroll_content.layout()

    def _on_int8_changed(self):
        """int8 勾选状态变化时更新 data 控件可见性"""
        self._update_params_visibility(self._current_format())

    def _on_precision_changed(self):
        self._update_params_visibility(self._current_format())

    def _current_format(self) -> str:
        return self._format_combo.currentData() or ""

    def _is_int8_selected(self, fmt: str = "") -> bool:
        fmt = fmt or self._current_format()
        if fmt == "engine":
            return self._precision_combo.currentData() == 8
        visible = FORMAT_PARAMS.get(fmt, set())
        return "int8" in visible and self._int8_check.isChecked()

    def _on_browse_data(self):
        """浏览选择校准数据集 YAML 文件"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择校准数据集 YAML", "", "YAML 文件 (*.yaml *.yml);;所有文件 (*)"
        )
        if path:
            self._data_edit.setText(path)
            self._update_int8_hint()

    def _update_int8_hint(self):
        """更新 INT8 提示标签的可见性"""
        fmt = self._current_format()
        data_empty = not self._data_edit.text().strip()
        self._int8_hint_label.setVisible(self._is_int8_selected(fmt) and data_empty)

    # ------------------------------------------------------------------
    def _on_preset_changed(self, preset_name: str):
        preset = EXPORT_PRESETS.get(preset_name)
        if preset is None:
            # 自定义预设时隐藏警告
            self._warning_label.setVisible(False)
            return
        # 设置格式
        fmt = preset.get("format", "onnx")
        format_index = self._format_combo.findData(fmt)
        if format_index >= 0:
            self._format_combo.setCurrentIndex(format_index)
        # 设置各参数
        for key, value in preset.items():
            if key == "format":
                continue
            widget = self._param_widgets.get(key)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif key == "precision" and isinstance(widget, QComboBox):
                index = widget.findData(int(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
        self._update_params_visibility(fmt)

        # Check task-preset mismatch
        if self._model_task and preset_name != "自定义":
            preset_task = None
            for t, p in TASK_PRESET_MAP.items():
                if p == preset_name:
                    preset_task = t
                    break
            if preset_task and preset_task != self._model_task:
                task_display = TASK_DISPLAY_NAMES.get(self._model_task, self._model_task)
                self._warning_label.setText(
                    f"⚠ 当前模型为{task_display}模型，选择此预设仅优化导出参数，不会改变模型任务类型"
                )
                self._warning_label.setVisible(True)
            else:
                self._warning_label.setVisible(False)
        elif preset_name == "自定义":
            self._warning_label.setVisible(False)

    # ------------------------------------------------------------------
    def _on_confirm(self):
        fmt = self._current_format()
        visible = FORMAT_PARAMS.get(fmt, set())
        kwargs = {}

        # imgsz -> int
        if "imgsz" in visible:
            kwargs["imgsz"] = int(self._imgsz_combo.currentText())

        # checkbox: only include if checked
        for name in ("half", "int8", "dynamic", "simplify", "nms"):
            if name in visible:
                widget = self._param_widgets[name]
                if widget.isChecked():
                    kwargs[name] = True

        if fmt == "engine":
            precision = self._precision_combo.currentData()
            if precision in (8, 16):
                kwargs["quantize"] = precision
            device = self._device_edit.text().strip()
            if not device:
                QMessageBox.warning(self, "TensorRT 设备缺失", "请输入 NVIDIA GPU 设备，例如 0。")
                return
            kwargs["device"] = device
            if precision == 8:
                kwargs["dynamic"] = True

        # INT8 calibration data is required for reproducible quantization.
        if self._is_int8_selected(fmt):
            data_path = self._data_edit.text().strip()
            if not data_path:
                QMessageBox.warning(
                    self, "INT8 量化校准数据缺失",
                    "INT8 量化需要校准数据集，通常使用训练数据的 data.yaml。\n"
                    "请选择校准数据文件后重试。",
                )
                self._confirm_btn.setEnabled(True)
                self._confirm_btn.setText("确认导出")
                return
            if not Path(data_path).exists():
                QMessageBox.warning(
                    self, "校准数据文件不存在",
                    f"校准数据文件不存在:\n{data_path}\n\n请选择有效的 data.yaml 文件。",
                )
                self._confirm_btn.setEnabled(True)
                self._confirm_btn.setText("确认导出")
                return
            if Path(data_path).suffix.lower() not in (".yaml", ".yml"):
                QMessageBox.warning(
                    self, "校准数据格式错误",
                    f"校准数据文件格式不正确: {Path(data_path).suffix}\n请选择 .yaml 或 .yml 文件。",
                )
                self._confirm_btn.setEnabled(True)
                self._confirm_btn.setText("确认导出")
                return
            kwargs["data"] = data_path
            if "fraction" in visible:
                kwargs["fraction"] = self._fraction_spin.value()

        # batch: only if > 1
        if "batch" in visible and self._batch_spin.value() > 1:
            kwargs["batch"] = self._batch_spin.value()

        # opset / workspace: always include if visible
        if "opset" in visible:
            kwargs["opset"] = self._opset_spin.value()
        if "workspace" in visible:
            workspace = self._workspace_spin.value()
            if workspace > 0:
                kwargs["workspace"] = workspace

        self.export_requested.emit(fmt, kwargs)

        self._confirm_btn.setEnabled(False)
        self._confirm_btn.setText("导出中...")
