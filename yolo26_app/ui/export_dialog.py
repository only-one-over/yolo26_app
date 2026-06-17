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
    "engine": {"imgsz", "half", "int8", "data", "dynamic", "batch", "workspace"},
    "coreml": {"imgsz", "half", "int8", "data", "dynamic", "batch"},
    "tflite": {"imgsz", "half", "int8", "data", "batch"},
    "ncnn": {"imgsz", "half", "batch"},
    "paddle": {"imgsz", "batch"},
    "mnn": {"imgsz", "half", "int8", "data", "batch"},
    "rknn": {"imgsz", "batch"},
}

EXPORT_PRESETS = {
    "自定义": None,
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
        self.setMinimumSize(500, 600)

        self._param_widgets = {}

        layout = QVBoxLayout(self)

        # --- 任务预设 ---
        preset_group = QGroupBox("任务预设")
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
        self._warning_label.setStyleSheet("color: #f9e2af; font-size: 12px;")
        self._warning_label.setVisible(False)
        preset_layout.addWidget(self._warning_label)

        layout.addWidget(preset_group)

        # --- 导出格式 ---
        format_group = QGroupBox("导出格式")
        format_layout = QHBoxLayout(format_group)
        self._format_combo = QComboBox()
        self._format_combo.addItems(list(FORMAT_PARAMS.keys()))
        self._format_combo.currentTextChanged.connect(self._update_params_visibility)
        format_layout.addWidget(self._format_combo)
        layout.addWidget(format_group)

        # --- 导出参数 ---
        params_group = QGroupBox("导出参数")
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

        # int8 hint label
        self._int8_hint_label = QLabel("INT8 量化需要校准数据集")
        self._int8_hint_label.setStyleSheet("color: #f9e2af; font-size: 12px;")
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

        # opset
        self._opset_spin = QSpinBox()
        self._opset_spin.setRange(9, 21)
        self._opset_spin.setValue(17)
        self._param_widgets["opset"] = self._opset_spin
        form.addRow("Opset", self._opset_spin)

        # workspace
        self._workspace_spin = QDoubleSpinBox()
        self._workspace_spin.setRange(1.0, 32.0)
        self._workspace_spin.setValue(4.0)
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

        scroll.setWidget(scroll_content)
        params_layout.addWidget(scroll)
        layout.addWidget(params_group)

        # --- 底部按钮 ---
        btn_layout = QHBoxLayout()
        self._confirm_btn = QPushButton("确认导出")
        self._confirm_btn.clicked.connect(self._on_confirm)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self._confirm_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # 初始可见性
        self._update_params_visibility(self._format_combo.currentText())

        # 根据模型任务自动选择预设
        if task in TASK_PRESET_MAP:
            self._preset_combo.setCurrentText(TASK_PRESET_MAP[task])

    # ------------------------------------------------------------------
    def _update_params_visibility(self, fmt: str):
        visible = FORMAT_PARAMS.get(fmt, set())
        for name, widget in self._param_widgets.items():
            if name == "data":
                # data 控件：仅当 int8 在可见参数中且被勾选时才显示
                data_visible = "int8" in visible and self._int8_check.isChecked()
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
        self._update_params_visibility(self._format_combo.currentText())

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
        fmt = self._format_combo.currentText()
        visible = FORMAT_PARAMS.get(fmt, set())
        int8_visible = "int8" in visible
        int8_checked = self._int8_check.isChecked()
        data_empty = not self._data_edit.text().strip()
        # 仅当 int8 可见且被勾选，且未选择 data 文件时显示提示
        self._int8_hint_label.setVisible(int8_visible and int8_checked and data_empty)

    # ------------------------------------------------------------------
    def _on_preset_changed(self, preset_name: str):
        preset = EXPORT_PRESETS.get(preset_name)
        if preset is None:
            # 自定义预设时隐藏警告
            self._warning_label.setVisible(False)
            return
        # 设置格式
        fmt = preset.get("format", "onnx")
        self._format_combo.setCurrentText(fmt)
        # 设置各参数
        for key, value in preset.items():
            if key == "format":
                continue
            widget = self._param_widgets.get(key)
            if widget is None:
                continue
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
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
        fmt = self._format_combo.currentText()
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

        # int8 data: if int8 is checked, data is required
        if "int8" in visible and self._int8_check.isChecked():
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

        # batch: only if > 1
        if "batch" in visible and self._batch_spin.value() > 1:
            kwargs["batch"] = self._batch_spin.value()

        # opset / workspace: always include if visible
        if "opset" in visible:
            kwargs["opset"] = self._opset_spin.value()
        if "workspace" in visible:
            kwargs["workspace"] = self._workspace_spin.value()

        self.export_requested.emit(fmt, kwargs)

        self._confirm_btn.setEnabled(False)
        self._confirm_btn.setText("导出中...")
