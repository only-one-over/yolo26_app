import glob
from pathlib import Path
from typing import List, Optional

import yaml
try:
    import pyqtgraph
    _PYQTGRAPH_AVAILABLE = True
except ImportError:
    _PYQTGRAPH_AVAILABLE = False

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QLabel,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QCheckBox,
    QFrame,
    QTabWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QCloseEvent, QPixmap

from yolo26_app.core.config import TrainConfig, ProjectConfig, normalize_augmentation_preset
from yolo26_app.core.logger import get_logger
from yolo26_app.core.model_registry import (
    MODEL_FAMILY_TASK_MODEL_MAP,
    AUGMENTATION_PRESET_LABELS,
    CUSTOM_AUGMENTATION_PRESET,
    AUGMENTATION_PRESET_ORDER,
    AUGMENTATION_PRESETS,
)
from yolo26_app.core.trainer import YOLOTrainer
from yolo26_app.ui import styles

logger = get_logger(__name__)

MODEL_FAMILY_MAP = {
    "YOLO26": "yolo26",
    "YOLOv8": "yolov8",
}

MODEL_INFO = {
    "n": "Nano | 3.2M 参数 | ≥2GB 显存 | 速度: ★★★★★",
    "s": "Small | 11.2M 参数 | ≥4GB 显存 | 速度: ★★★★",
    "m": "Medium | 25.9M 参数 | ≥8GB 显存 | 速度: ★★★",
    "l": "Large | 43.7M 参数 | ≥12GB 显存 | 速度: ★★",
    "x": "XLarge | 68.4M 参数 | ≥16GB 显存 | 速度: ★",
}

TASK_INFO = {
    "detect": "目标检测 — 检测图中的目标并给出矩形框",
    "segment": "实例分割 — 检测目标并生成精确像素掩码",
    "classify": "图像分类 — 对整张图片进行类别分类",
    "pose": "姿态估计 — 检测人体关键点和骨架",
}


def parse_results_csv(csv_path: Path) -> dict:
    """解析 Ultralytics results.csv,返回 {列名: [值]} 字典。跳过空行与不完整行。"""
    import csv as _csv
    if not csv_path.exists():
        return {}
    columns: dict = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(_csv.reader(f))
            # Ultralytics 的 CSV 表头可能包含前导空格(如 " train/box_loss"),需 strip
            fieldnames = [fn.strip() for fn in reader.fieldnames or []]
            for fn in fieldnames:
                columns[fn] = []
            for row in reader:
                # 检查行是否完整(所有字段都有值且可转 float)
                try:
                    parsed_row = {}
                    for fn in fieldnames:
                        val = (row.get(fn) or "").strip()
                        if not val:
                            raise ValueError("incomplete row")
                        parsed_row[fn] = float(val)
                    for fn, v in parsed_row.items():
                        columns[fn].append(v)
                except (ValueError, TypeError):
                    continue
    except (OSError, _csv.Error):
        return {}
    return columns


class TrainWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._trainer: Optional[YOLOTrainer] = None
        self._project_path: str = ""
        self._setup_ui()
        self._csv_timer = QTimer(self)
        self._csv_timer.setInterval(5000)
        self._csv_timer.timeout.connect(self._refresh_curves)
        self._current_save_dir: Optional[Path] = None

    def _setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left panel: configuration (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(12, 12, 8, 12)
        left_layout.setSpacing(8)

        # Right panel: monitoring (fixed min width)
        right_panel = QWidget()
        right_panel.setMinimumWidth(360)
        right_panel.setObjectName("trainMonitorPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 12, 12, 12)
        right_layout.setSpacing(8)

        config_group = QGroupBox("基本配置")
        config_group.setObjectName("configCard")
        form = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "classify", "pose"])
        form.addRow("任务类型:", self.task_combo)

        self.family_combo = QComboBox()
        self.family_combo.addItems(list(MODEL_FAMILY_MAP.keys()))
        form.addRow("模型系列:", self.family_combo)

        self._task_info_label = QLabel(TASK_INFO.get("detect", ""))
        self._task_info_label.setObjectName("infoLabel")
        self._task_info_label.setWordWrap(True)
        form.addRow("", self._task_info_label)

        self.size_combo = QComboBox()
        self.size_combo.addItems(["n", "s", "m", "l", "x"])
        form.addRow("模型大小:", self.size_combo)

        custom_model_row = QHBoxLayout()
        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("可选：自定义 .pt 或 .yaml 模型路径")
        custom_model_browse = QPushButton("浏览")
        custom_model_browse.setMinimumWidth(60)
        custom_model_browse.clicked.connect(self._browse_custom_model)
        custom_model_row.addWidget(self.custom_model_edit)
        custom_model_row.addWidget(custom_model_browse)
        form.addRow("自定义模型:", custom_model_row)

        self._model_preview_label = QLabel()
        self._model_preview_label.setObjectName("infoLabel")
        self._model_preview_label.setWordWrap(True)
        form.addRow("", self._model_preview_label)

        self._model_info_label = QLabel(MODEL_INFO.get("n", ""))
        self._model_info_label.setObjectName("infoLabel")
        self._model_info_label.setWordWrap(True)
        form.addRow("", self._model_info_label)

        data_row = QHBoxLayout()
        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("选择数据集 .yaml 文件")
        data_browse = QPushButton("浏览")
        data_browse.setMinimumWidth(60)
        data_browse.clicked.connect(self._browse_dataset)
        data_row.addWidget(self.data_edit)
        data_row.addWidget(data_browse)
        form.addRow("数据集路径:", data_row)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        form.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        form.addRow("Batch Size:", self.batch_spin)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "480", "640", "960", "1280"])
        self.imgsz_combo.setCurrentText("640")
        form.addRow("图像尺寸:", self.imgsz_combo)

        self.device_edit = QLineEdit()
        self.device_edit.setPlaceholderText("auto/cpu/0/0,1")
        form.addRow("设备:", self.device_edit)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW"])
        form.addRow("优化器:", self.optimizer_combo)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        form.addRow("学习率:", self.lr_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(100)
        form.addRow("早停耐心:", self.patience_spin)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("train")
        form.addRow("实验名称:", self.name_edit)

        config_group.setLayout(form)
        left_layout.addWidget(config_group)

        # 高级设置组（可折叠）
        self._advanced_group = QGroupBox("高级设置")
        self._advanced_group.setObjectName("configCard")
        self._advanced_group.setCheckable(True)
        self._advanced_group.setChecked(False)
        advanced_form = QFormLayout()

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(8)
        advanced_form.addRow("Workers:", self.workers_spin)

        self.cache_check = QCheckBox()
        self.cache_check.setChecked(False)
        advanced_form.addRow("缓存数据:", self.cache_check)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(0)
        advanced_form.addRow("随机种子:", self.seed_spin)

        self.plots_check = QCheckBox()
        self.plots_check.setChecked(True)
        advanced_form.addRow("生成图表:", self.plots_check)

        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 100)
        self.close_mosaic_spin.setValue(10)

        # 数据增强分组
        aug_group = QGroupBox("数据增强")
        aug_group.setObjectName("configCard")
        aug_layout = QFormLayout(aug_group)

        self.aug_enabled_check = QCheckBox()
        self.aug_enabled_check.setChecked(True)
        aug_layout.addRow("启用数据增强:", self.aug_enabled_check)

        self.aug_preset_combo = QComboBox()
        for preset_key in AUGMENTATION_PRESET_ORDER:
            self.aug_preset_combo.addItem(AUGMENTATION_PRESET_LABELS[preset_key], preset_key)
        self.aug_preset_combo.addItem(AUGMENTATION_PRESET_LABELS[CUSTOM_AUGMENTATION_PRESET], CUSTOM_AUGMENTATION_PRESET)
        self.aug_preset_combo.setCurrentIndex(self.aug_preset_combo.findData("default"))
        aug_layout.addRow("增强预设:", self.aug_preset_combo)

        self._augmentation_task_hint_label = QLabel()
        self._augmentation_task_hint_label.setObjectName("infoLabel")
        self._augmentation_task_hint_label.setWordWrap(True)
        aug_layout.addRow("", self._augmentation_task_hint_label)

        # 高级增强参数（可折叠）
        self._aug_advanced_widget = QWidget()
        aug_advanced_layout = QVBoxLayout(self._aug_advanced_widget)
        aug_advanced_layout.setContentsMargins(0, 0, 0, 0)

        color_group = QGroupBox("颜色增强")
        color_form = QFormLayout(color_group)

        self.hsv_h_spin = QDoubleSpinBox(); self.hsv_h_spin.setRange(0, 1); self.hsv_h_spin.setDecimals(3); self.hsv_h_spin.setSingleStep(0.005); self.hsv_h_spin.setValue(0.015)
        color_form.addRow("HSV 色调:", self.hsv_h_spin)

        self.hsv_s_spin = QDoubleSpinBox(); self.hsv_s_spin.setRange(0, 1); self.hsv_s_spin.setDecimals(2); self.hsv_s_spin.setSingleStep(0.05); self.hsv_s_spin.setValue(0.7)
        color_form.addRow("HSV 饱和度:", self.hsv_s_spin)

        self.hsv_v_spin = QDoubleSpinBox(); self.hsv_v_spin.setRange(0, 1); self.hsv_v_spin.setDecimals(2); self.hsv_v_spin.setSingleStep(0.05); self.hsv_v_spin.setValue(0.4)
        color_form.addRow("HSV 亮度:", self.hsv_v_spin)

        geometry_group = QGroupBox("几何增强")
        geometry_form = QFormLayout(geometry_group)

        self.degrees_spin = QDoubleSpinBox(); self.degrees_spin.setRange(0, 180); self.degrees_spin.setDecimals(1); self.degrees_spin.setValue(0)
        geometry_form.addRow("旋转角度:", self.degrees_spin)

        self.translate_spin = QDoubleSpinBox(); self.translate_spin.setRange(0, 1); self.translate_spin.setDecimals(2); self.translate_spin.setSingleStep(0.05); self.translate_spin.setValue(0.1)
        geometry_form.addRow("平移:", self.translate_spin)

        self.scale_spin = QDoubleSpinBox(); self.scale_spin.setRange(0, 1); self.scale_spin.setDecimals(2); self.scale_spin.setSingleStep(0.05); self.scale_spin.setValue(0.5)
        geometry_form.addRow("缩放:", self.scale_spin)

        self.shear_spin = QDoubleSpinBox(); self.shear_spin.setRange(0, 180); self.shear_spin.setDecimals(1); self.shear_spin.setValue(0)
        geometry_form.addRow("剪切:", self.shear_spin)

        self.perspective_spin = QDoubleSpinBox(); self.perspective_spin.setRange(0, 0.01); self.perspective_spin.setDecimals(4); self.perspective_spin.setSingleStep(0.0005); self.perspective_spin.setValue(0)
        geometry_form.addRow("透视:", self.perspective_spin)

        self.flipud_spin = QDoubleSpinBox(); self.flipud_spin.setRange(0, 1); self.flipud_spin.setDecimals(2); self.flipud_spin.setSingleStep(0.05); self.flipud_spin.setValue(0)
        geometry_form.addRow("上下翻转:", self.flipud_spin)

        self.fliplr_spin = QDoubleSpinBox(); self.fliplr_spin.setRange(0, 1); self.fliplr_spin.setDecimals(2); self.fliplr_spin.setSingleStep(0.05); self.fliplr_spin.setValue(0.5)
        geometry_form.addRow("左右翻转:", self.fliplr_spin)

        composite_group = QGroupBox("组合增强")
        composite_form = QFormLayout(composite_group)

        self.mosaic_spin = QDoubleSpinBox(); self.mosaic_spin.setRange(0, 1); self.mosaic_spin.setDecimals(2); self.mosaic_spin.setSingleStep(0.1); self.mosaic_spin.setValue(1.0)
        composite_form.addRow("Mosaic:", self.mosaic_spin)

        self.mixup_spin = QDoubleSpinBox(); self.mixup_spin.setRange(0, 1); self.mixup_spin.setDecimals(2); self.mixup_spin.setSingleStep(0.05); self.mixup_spin.setValue(0)
        composite_form.addRow("MixUp:", self.mixup_spin)

        self.cutmix_spin = QDoubleSpinBox(); self.cutmix_spin.setRange(0, 1); self.cutmix_spin.setDecimals(2); self.cutmix_spin.setSingleStep(0.05); self.cutmix_spin.setValue(0)
        composite_form.addRow("CutMix:", self.cutmix_spin)

        self.copy_paste_spin = QDoubleSpinBox(); self.copy_paste_spin.setRange(0, 1); self.copy_paste_spin.setDecimals(2); self.copy_paste_spin.setSingleStep(0.05); self.copy_paste_spin.setValue(0)
        composite_form.addRow("Copy-Paste:", self.copy_paste_spin)

        composite_form.addRow("最后 N 轮关闭:", self.close_mosaic_spin)

        expert_group = QGroupBox("高级增强")
        expert_form = QFormLayout(expert_group)

        self.erasing_spin = QDoubleSpinBox(); self.erasing_spin.setRange(0, 1); self.erasing_spin.setDecimals(2); self.erasing_spin.setSingleStep(0.05); self.erasing_spin.setValue(0.4)
        expert_form.addRow("擦除:", self.erasing_spin)

        self.auto_augment_combo = QComboBox()
        self.auto_augment_combo.addItems(["randaugment", "autoaugment", ""])
        expert_form.addRow("自动增强:", self.auto_augment_combo)

        expert_hint = QLabel("提示：擦除和自动增强主要用于分类增强或新版 Ultralytics 支持场景。")
        expert_hint.setObjectName("infoLabel")
        expert_hint.setWordWrap(True)
        expert_form.addRow("", expert_hint)

        aug_advanced_layout.addWidget(color_group)
        aug_advanced_layout.addWidget(geometry_group)
        aug_advanced_layout.addWidget(composite_group)
        aug_advanced_layout.addWidget(expert_group)

        aug_layout.addRow(self._aug_advanced_widget)

        self._advanced_group.setLayout(advanced_form)
        left_layout.addWidget(self._advanced_group)
        left_layout.addWidget(aug_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_container)
        main_layout.addWidget(left_scroll, 1)

        self.size_combo.currentTextChanged.connect(self._update_model_info)
        self.task_combo.currentTextChanged.connect(self._update_task_info)
        self.family_combo.currentTextChanged.connect(self._update_model_preview)
        self.task_combo.currentTextChanged.connect(self._update_model_preview)
        self.size_combo.currentTextChanged.connect(self._update_model_preview)
        self.custom_model_edit.textChanged.connect(self._update_model_preview)

        self.aug_enabled_check.toggled.connect(self._on_aug_enabled_changed)
        self.aug_preset_combo.currentTextChanged.connect(self._on_aug_preset_changed)
        # 增强参数变化时检测是否为自定义
        for spin in [self.hsv_h_spin, self.hsv_s_spin, self.hsv_v_spin,
                     self.degrees_spin, self.translate_spin, self.scale_spin,
                     self.shear_spin, self.perspective_spin, self.flipud_spin,
                     self.fliplr_spin, self.mosaic_spin, self.mixup_spin,
                     self.cutmix_spin, self.copy_paste_spin, self.erasing_spin]:
            spin.valueChanged.connect(self._on_aug_param_changed)
        self.auto_augment_combo.currentTextChanged.connect(self._on_aug_param_changed)

        self._update_model_preview()
        self._update_task_info(self.task_combo.currentText())

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        right_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        right_layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setObjectName("logView")
        self.log_text.setMinimumHeight(200)
        right_layout.addWidget(self.log_text)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setStyleSheet(styles.START_BUTTON_STYLE)
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(styles.STOP_BUTTON_STYLE)
        self.stop_btn.clicked.connect(self._on_stop)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        right_layout.addLayout(btn_layout)

        # ===== 训练曲线可视化面板 =====
        self.curves_tab = QTabWidget(self)
        self.curves_tab.setMinimumHeight(350)

        # Loss 标签页
        if _PYQTGRAPH_AVAILABLE:
            self.loss_plot = pyqtgraph.PlotWidget(self)
            self.loss_plot.setLabel("left", "loss")
            self.loss_plot.setLabel("bottom", "epoch")
            self.loss_plot.addLegend()
            self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
            self.curves_tab.addTab(self.loss_plot, "Loss 曲线")
        else:
            self.loss_plot = None
            loss_placeholder = QLabel("请安装 pyqtgraph 以启用训练曲线:\n  pip install pyqtgraph")
            loss_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.curves_tab.addTab(loss_placeholder, "Loss 曲线")

        # mAP 标签页
        if _PYQTGRAPH_AVAILABLE:
            self.map_plot = pyqtgraph.PlotWidget(self)
            self.map_plot.setLabel("left", "mAP")
            self.map_plot.setLabel("bottom", "epoch")
            self.map_plot.addLegend()
            self.map_plot.showGrid(x=True, y=True, alpha=0.3)
            self.curves_tab.addTab(self.map_plot, "mAP 曲线")
        else:
            self.map_plot = None
            map_placeholder = QLabel("请安装 pyqtgraph 以启用训练曲线:\n  pip install pyqtgraph")
            map_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.curves_tab.addTab(map_placeholder, "mAP 曲线")

        # PR/F1/P/R 标签页
        pr_scroll = QScrollArea(self)
        pr_scroll.setWidgetResizable(True)
        pr_container = QWidget()
        pr_layout = QVBoxLayout(pr_container)
        self.pr_label = QLabel("训练完成后显示 PR 曲线")
        self.f1_label = QLabel("训练完成后显示 F1 曲线")
        self.p_label = QLabel("训练完成后显示 P 曲线")
        self.r_label = QLabel("训练完成后显示 R 曲线")
        for lbl in (self.pr_label, self.f1_label, self.p_label, self.r_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pr_layout.addWidget(lbl)
        pr_scroll.setWidget(pr_container)
        self.curves_tab.addTab(pr_scroll, "PR / F1 / P / R")

        # 混淆矩阵标签页
        cm_scroll = QScrollArea(self)
        cm_scroll.setWidgetResizable(True)
        cm_container = QWidget()
        cm_layout = QVBoxLayout(cm_container)
        self.cm_label = QLabel("训练完成后显示混淆矩阵")
        self.cm_norm_label = QLabel("训练完成后显示归一化混淆矩阵")
        for lbl in (self.cm_label, self.cm_norm_label):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cm_layout.addWidget(lbl)
        cm_scroll.setWidget(cm_container)
        self.curves_tab.addTab(cm_scroll, "混淆矩阵")

        right_layout.addWidget(self.curves_tab)

        self.results_group = QGroupBox("训练结果")
        results_layout = QVBoxLayout()
        self.result_model_label = QLabel("最佳模型: -")
        self.result_metrics_label = QLabel("指标: -")
        results_layout.addWidget(self.result_model_label)
        results_layout.addWidget(self.result_metrics_label)
        self.results_group.setLayout(results_layout)
        self.results_group.hide()
        right_layout.addWidget(self.results_group)

        # 打开 runs 目录按钮
        self.open_runs_btn = QPushButton("打开 runs 目录", self)
        self.open_runs_btn.clicked.connect(self._on_open_runs)
        right_layout.addWidget(self.open_runs_btn)

        main_layout.addWidget(right_panel, 0)

    def _update_model_info(self, size: str) -> None:
        self._model_info_label.setText(MODEL_INFO.get(size, ""))

    def _get_project_subdir(self, subdir: str) -> str:
        """返回工作区间下指定子目录路径,若不存在则回退到用户主目录。"""
        if self._project_path:
            candidate = Path(self._project_path) / subdir
            if candidate.is_dir():
                return str(candidate)
        return str(Path.home())

    def _browse_custom_model(self) -> None:
        from yolo26_app.core.paths import SYSTEM_MODEL_SUBDIRS
        start_dir = str(Path.home())
        yolo_model_dir = SYSTEM_MODEL_SUBDIRS["yolo"]
        if yolo_model_dir.is_dir():
            start_dir = str(yolo_model_dir)
        elif self._project_path:
            models_dir = Path(self._project_path) / "models"
            if models_dir.is_dir():
                start_dir = str(models_dir)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择自定义模型", start_dir, "模型文件 (*.pt *.yaml *.yml);;所有文件 (*)"
        )
        if path:
            self.custom_model_edit.setText(path)

    def _update_model_preview(self) -> None:
        custom = self.custom_model_edit.text().strip()
        if custom:
            self._model_preview_label.setText(f"将加载模型：{custom}")
            return
        family_key = MODEL_FAMILY_MAP.get(self.family_combo.currentText(), "yolo26")
        family_map = MODEL_FAMILY_TASK_MODEL_MAP.get(family_key, MODEL_FAMILY_TASK_MODEL_MAP["yolo26"])
        task = self.task_combo.currentText()
        size = self.size_combo.currentText()
        template = family_map.get(task, family_map["detect"])
        model_name = template.format(size=size)
        self._model_preview_label.setText(f"将加载模型：{model_name}")

    def _on_aug_enabled_changed(self, checked: bool) -> None:
        """启用/禁用数据增强时切换控件可见性"""
        self.aug_preset_combo.setVisible(checked)
        self._augmentation_task_hint_label.setVisible(checked)
        self._aug_advanced_widget.setVisible(checked)

    def _augmentation_spin_widgets(self) -> List[QDoubleSpinBox]:
        return [
            self.hsv_h_spin,
            self.hsv_s_spin,
            self.hsv_v_spin,
            self.degrees_spin,
            self.translate_spin,
            self.scale_spin,
            self.shear_spin,
            self.perspective_spin,
            self.flipud_spin,
            self.fliplr_spin,
            self.mosaic_spin,
            self.mixup_spin,
            self.cutmix_spin,
            self.copy_paste_spin,
            self.erasing_spin,
        ]

    def _current_aug_preset_key(self) -> str:
        data = self.aug_preset_combo.currentData()
        return normalize_augmentation_preset(data or self.aug_preset_combo.currentText())

    def _set_aug_preset_key(self, preset_key: str) -> None:
        normalized = normalize_augmentation_preset(preset_key)
        index = self.aug_preset_combo.findData(normalized)
        if index < 0:
            index = self.aug_preset_combo.findData("default")
        self.aug_preset_combo.setCurrentIndex(index)

    def _set_augmentation_values(self, preset: dict) -> None:
        for spin in self._augmentation_spin_widgets():
            spin.blockSignals(True)
        self.auto_augment_combo.blockSignals(True)

        self.hsv_h_spin.setValue(preset["hsv_h"])
        self.hsv_s_spin.setValue(preset["hsv_s"])
        self.hsv_v_spin.setValue(preset["hsv_v"])
        self.degrees_spin.setValue(preset["degrees"])
        self.translate_spin.setValue(preset["translate"])
        self.scale_spin.setValue(preset["scale"])
        self.shear_spin.setValue(preset["shear"])
        self.perspective_spin.setValue(preset["perspective"])
        self.flipud_spin.setValue(preset["flipud"])
        self.fliplr_spin.setValue(preset["fliplr"])
        self.mosaic_spin.setValue(preset["mosaic"])
        self.mixup_spin.setValue(preset["mixup"])
        self.cutmix_spin.setValue(preset["cutmix"])
        self.copy_paste_spin.setValue(preset["copy_paste"])
        self.erasing_spin.setValue(preset["erasing"])
        self.auto_augment_combo.setCurrentText(preset["auto_augment"])

        for spin in self._augmentation_spin_widgets():
            spin.blockSignals(False)
        self.auto_augment_combo.blockSignals(False)

    def _current_augmentation_values(self) -> dict:
        return {
            "hsv_h": self.hsv_h_spin.value(),
            "hsv_s": self.hsv_s_spin.value(),
            "hsv_v": self.hsv_v_spin.value(),
            "degrees": self.degrees_spin.value(),
            "translate": self.translate_spin.value(),
            "scale": self.scale_spin.value(),
            "shear": self.shear_spin.value(),
            "perspective": self.perspective_spin.value(),
            "flipud": self.flipud_spin.value(),
            "fliplr": self.fliplr_spin.value(),
            "mosaic": self.mosaic_spin.value(),
            "mixup": self.mixup_spin.value(),
            "cutmix": self.cutmix_spin.value(),
            "copy_paste": self.copy_paste_spin.value(),
            "erasing": self.erasing_spin.value(),
            "auto_augment": self.auto_augment_combo.currentText(),
        }

    def _on_aug_preset_changed(self, preset_name: str) -> None:
        """切换增强预设时更新参数值"""
        preset_key = self._current_aug_preset_key()
        if preset_key == CUSTOM_AUGMENTATION_PRESET:
            return
        preset = AUGMENTATION_PRESETS.get(preset_key)
        if preset is None:
            return
        self._set_augmentation_values(preset)
        self._apply_task_augmentation_advice()

    def _on_aug_param_changed(self) -> None:
        """手动修改参数时将预设改为自定义"""
        current = self._current_aug_preset_key()
        if current == CUSTOM_AUGMENTATION_PRESET:
            return
        preset = AUGMENTATION_PRESETS.get(current)
        if preset is None:
            self._set_aug_preset_key(CUSTOM_AUGMENTATION_PRESET)
            return
        values = self._current_augmentation_values()
        if any(values[key] != preset[key] for key in preset):
            self._set_aug_preset_key(CUSTOM_AUGMENTATION_PRESET)

    def _update_task_info(self, task: str) -> None:
        self._task_info_label.setText(TASK_INFO.get(task, ""))
        self._apply_task_augmentation_advice()

    def _apply_task_augmentation_advice(self) -> None:
        task = self.task_combo.currentText()
        if task == "pose":
            self._augmentation_task_hint_label.setText(
                "姿态任务建议保持上下翻转 flipud=0，避免关键点左右/上下语义被破坏。"
            )
            if self.flipud_spin.value() != 0:
                self.flipud_spin.blockSignals(True)
                self.flipud_spin.setValue(0)
                self.flipud_spin.blockSignals(False)
                self._set_aug_preset_key(CUSTOM_AUGMENTATION_PRESET)
        elif task == "segment":
            self._augmentation_task_hint_label.setText(
                "分割任务建议保守使用 Copy-Paste，默认保持 0.0；需要强增强时再手动提高。"
            )
        elif task == "classify":
            self._augmentation_task_hint_label.setText(
                "分类任务可重点关注 erasing 和 auto_augment；检测类组合增强仍由 Ultralytics 处理。"
            )
        else:
            self._augmentation_task_hint_label.setText(
                "检测任务可优先调整 Mosaic、MixUp、CutMix、HSV、缩放和左右翻转。"
            )

    def _browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置文件", self._get_project_subdir("datasets"),
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.data_edit.setText(path)

    def _persist_train_config(self, config: TrainConfig) -> None:
        """将训练配置保存到项目配置文件"""
        window = self.window()
        if not hasattr(window, "current_project_config"):
            return
        project_config = window.current_project_config
        if project_config is None:
            return
        project_config.train_config = config
        try:
            from yolo26_app.core.project_manager import ProjectManager
            config_path = Path(project_config.project_path) / ProjectManager.CONFIG_FILENAME
            project_config.save(config_path)
        except Exception:
            pass

    def _build_config(self) -> TrainConfig:
        return TrainConfig(
            task=self.task_combo.currentText(),
            model_size=self.size_combo.currentText(),
            data=self.data_edit.text().strip(),
            epochs=self.epochs_spin.value(),
            batch=self.batch_spin.value(),
            imgsz=int(self.imgsz_combo.currentText()),
            device=self.device_edit.text().strip(),
            optimizer=self.optimizer_combo.currentText(),
            lr0=self.lr_spin.value(),
            patience=self.patience_spin.value(),
            name=self.name_edit.text().strip(),
            workers=self.workers_spin.value(),
            cache=self.cache_check.isChecked(),
            seed=self.seed_spin.value(),
            plots=self.plots_check.isChecked(),
            close_mosaic=self.close_mosaic_spin.value(),
            model_family=MODEL_FAMILY_MAP.get(self.family_combo.currentText(), "yolo26"),
            pretrained_model=self.custom_model_edit.text().strip(),
            augmentation_enabled=self.aug_enabled_check.isChecked(),
            augmentation_preset=self._current_aug_preset_key(),
            hsv_h=self.hsv_h_spin.value(),
            hsv_s=self.hsv_s_spin.value(),
            hsv_v=self.hsv_v_spin.value(),
            degrees=self.degrees_spin.value(),
            translate=self.translate_spin.value(),
            scale=self.scale_spin.value(),
            shear=self.shear_spin.value(),
            perspective=self.perspective_spin.value(),
            flipud=self.flipud_spin.value(),
            fliplr=self.fliplr_spin.value(),
            mosaic=self.mosaic_spin.value(),
            mixup=self.mixup_spin.value(),
            cutmix=self.cutmix_spin.value(),
            copy_paste=self.copy_paste_spin.value(),
            erasing=self.erasing_spin.value(),
            auto_augment=self.auto_augment_combo.currentText(),
        )

    def _set_form_enabled(self, enabled: bool) -> None:
        self.task_combo.setEnabled(enabled)
        self.size_combo.setEnabled(enabled)
        self.data_edit.setEnabled(enabled)
        self.epochs_spin.setEnabled(enabled)
        self.batch_spin.setEnabled(enabled)
        self.imgsz_combo.setEnabled(enabled)
        self.device_edit.setEnabled(enabled)
        self.optimizer_combo.setEnabled(enabled)
        self.lr_spin.setEnabled(enabled)
        self.patience_spin.setEnabled(enabled)
        self.name_edit.setEnabled(enabled)
        self.workers_spin.setEnabled(enabled)
        self.cache_check.setEnabled(enabled)
        self.seed_spin.setEnabled(enabled)
        self.plots_check.setEnabled(enabled)
        self.close_mosaic_spin.setEnabled(enabled)
        self.family_combo.setEnabled(enabled)
        self.custom_model_edit.setEnabled(enabled)
        self.aug_enabled_check.setEnabled(enabled)
        self.aug_preset_combo.setEnabled(enabled)
        self.hsv_h_spin.setEnabled(enabled)
        self.hsv_s_spin.setEnabled(enabled)
        self.hsv_v_spin.setEnabled(enabled)
        self.degrees_spin.setEnabled(enabled)
        self.translate_spin.setEnabled(enabled)
        self.scale_spin.setEnabled(enabled)
        self.shear_spin.setEnabled(enabled)
        self.perspective_spin.setEnabled(enabled)
        self.flipud_spin.setEnabled(enabled)
        self.fliplr_spin.setEnabled(enabled)
        self.mosaic_spin.setEnabled(enabled)
        self.mixup_spin.setEnabled(enabled)
        self.cutmix_spin.setEnabled(enabled)
        self.copy_paste_spin.setEnabled(enabled)
        self.erasing_spin.setEnabled(enabled)
        self.auto_augment_combo.setEnabled(enabled)

    def _validate_dataset(self) -> bool:
        data_path = self.data_edit.text().strip()
        if not data_path or not Path(data_path).exists():
            QMessageBox.warning(self, "验证失败", f"数据集配置文件不存在: {data_path}")
            return False
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.warning(self, "验证失败", f"无法读取数据集配置: {e}")
            return False

        data_yaml_path = Path(data_path).resolve()
        raw_path = cfg.get("path", "")
        if raw_path:
            if Path(raw_path).is_absolute():
                base = Path(raw_path).resolve()
            else:
                base = (data_yaml_path.parent / raw_path).resolve()
        else:
            base = data_yaml_path.parent

        task = self.task_combo.currentText()
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

        if task in ("detect", "segment", "pose"):
            for key in ("train", "val"):
                img_dir = (base / cfg.get(key, f"images/{key}")).resolve()
                lbl_dir = img_dir.parent.parent / "labels" / key
                if not img_dir.exists():
                    QMessageBox.warning(self, "验证失败", f"任务类型: {task}\n目录不存在: {img_dir}")
                    return False
                if not lbl_dir.exists():
                    QMessageBox.warning(self, "验证失败", f"任务类型: {task}\n目录不存在: {lbl_dir}")
                    return False
                images = [
                    p for p in glob.glob(str(img_dir / "**" / "*.*"), recursive=True)
                    if Path(p).suffix.lower() in img_exts
                ]
                if not images:
                    QMessageBox.warning(self, "验证失败", f"任务类型: {task}\n目录中没有图片: {img_dir}")
                    return False
        elif task == "classify":
            for key in ("train", "val"):
                split_dir = (base / cfg.get(key, key)).resolve()
                if not split_dir.exists():
                    QMessageBox.warning(self, "验证失败", f"任务类型: {task}\n目录不存在: {split_dir}")
                    return False
                images = [
                    p for p in glob.glob(str(split_dir / "**" / "*.*"), recursive=True)
                    if Path(p).suffix.lower() in img_exts
                ]
                if not images:
                    QMessageBox.warning(self, "验证失败", f"任务类型: {task}\n目录中没有图片: {split_dir}")
                    return False
        return True

    def _on_start(self) -> None:
        if not self.data_edit.text().strip():
            QMessageBox.warning(self, "验证失败", "请先选择数据集配置文件 (.yaml)")
            return

        if not self._validate_dataset():
            return

        config = self._build_config()
        project_path = self._project_path or ""

        # 训练前保存配置到项目
        self._persist_train_config(config)

        self._trainer = YOLOTrainer(config, project_path)
        self._trainer.progress_signal.connect(self._on_progress)
        self._trainer.log_signal.connect(self._on_log)
        self._trainer.finished_signal.connect(self._on_finished)
        self._trainer.error_signal.connect(self._on_error)

        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("训练中...")
        self.results_group.hide()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_form_enabled(False)

        save_dir = Path(self._project_path) / "runs" / (config.name or "train")
        self._current_save_dir = save_dir
        self._csv_timer.start()

        self._trainer.start()

    def _on_stop(self) -> None:
        if self._trainer and self._trainer.isRunning():
            self._trainer.stop()
            self.status_label.setText("正在停止训练...")
            self.stop_btn.setEnabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._csv_timer.stop()
        if self._trainer is not None and self._trainer.isRunning():
            self._on_stop()
            self._trainer.wait(30000)
            if self._trainer.isRunning():
                logger.warning("警告:训练线程未在 30 秒内退出,可能仍在后台运行")
        super().closeEvent(event)

    def _on_progress(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"训练中: Epoch {current}/{total}")

    def _on_log(self, message: str) -> None:
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _refresh_curves(self) -> None:
        if not _PYQTGRAPH_AVAILABLE:
            return
        if self._current_save_dir is None:
            return
        csv_path = self._current_save_dir / "results.csv"
        columns = parse_results_csv(csv_path)
        if not columns:
            return
        epochs = columns.get("epoch", list(range(1, len(next(iter(columns.values()))) + 1)))
        # Loss 曲线
        self.loss_plot.clear()
        loss_columns = [
            ("train/box_loss", "train/box", "#1f77b4"),
            ("train/cls_loss", "train/cls", "#ff7f0e"),
            ("train/seg_loss", "train/seg", "#2ca02c"),
            ("val/box_loss", "val/box", "#d62728"),
            ("val/cls_loss", "val/cls", "#9467bd"),
            ("val/seg_loss", "val/seg", "#8c564b"),
        ]
        for col_name, legend_name, color in loss_columns:
            if col_name in columns:
                self.loss_plot.plot(epochs, columns[col_name], pen=pyqtgraph.mkPen(color, width=2), name=legend_name)
        # mAP 曲线
        self.map_plot.clear()
        map_columns = [
            ("metrics/mAP50(B)", "mAP50", "#1f77b4"),
            ("metrics/mAP50-95(B)", "mAP50-95", "#ff7f0e"),
        ]
        for col_name, legend_name, color in map_columns:
            if col_name in columns:
                self.map_plot.plot(epochs, columns[col_name], pen=pyqtgraph.mkPen(color, width=2), name=legend_name)

    def _on_finished(self, message: str) -> None:
        self._csv_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_form_enabled(True)
        self.status_label.setText("训练完成")

        lines = message.split("\n")
        model_path = ""
        metrics_lines: List[str] = []
        in_metrics = False

        for line in lines:
            if line.startswith("最佳模型:"):
                model_path = line.replace("最佳模型:", "").strip()
                self.result_model_label.setText(f"最佳模型: {model_path}")
            elif line.strip() == "指标:":
                in_metrics = True
            elif in_metrics and line.strip():
                metrics_lines.append(line.strip())

        if metrics_lines:
            self.result_metrics_label.setText("指标:\n" + "\n".join(metrics_lines))
        else:
            self.result_metrics_label.setText("指标: 训练已完成")

        self.results_group.show()

        # 加载 PNG 图表
        if self._current_save_dir is not None:
            save_dir = self._current_save_dir
            # PR/F1/P/R
            png_map = [
                (self.pr_label, "PR_curve.png", "PR 曲线"),
                (self.f1_label, "F1_curve.png", "F1 曲线"),
                (self.p_label, "P_curve.png", "P 曲线"),
                (self.r_label, "R_curve.png", "R 曲线"),
                (self.cm_label, "confusion_matrix.png", "混淆矩阵"),
                (self.cm_norm_label, "confusion_matrix_normalized.png", "归一化混淆矩阵"),
            ]
            for label, filename, default_text in png_map:
                png_path = save_dir / filename
                if png_path.exists():
                    pixmap = QPixmap(str(png_path))
                    if not pixmap.isNull():
                        # 缩放以适应标签宽度(保持比例)
                        scaled = pixmap.scaledToWidth(
                            max(label.width(), 600),
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        label.setPixmap(scaled)
                    else:
                        label.setText(f"{default_text} - 图表加载失败")
                else:
                    label.setText(f"无此图表: {filename}")
            # 最终刷新一次曲线
            self._refresh_curves()

    def _on_error(self, message: str) -> None:
        self._csv_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_form_enabled(True)
        self.status_label.setText("训练出错")
        QMessageBox.critical(self, "训练错误", message)

    def _on_open_runs(self) -> None:
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        if not self._project_path:
            QMessageBox.information(self, "提示", "请先选择项目")
            return
        runs_dir = Path(self._project_path) / "runs"
        if not runs_dir.exists():
            QMessageBox.information(self, "提示", "暂无训练记录")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(runs_dir)))

    def set_project_config(self, config: Optional[ProjectConfig]) -> None:
        if config is None:
            # 自由空间模式:清空表单
            self._project_path = ""
            self.data_edit.setText("")
            self.custom_model_edit.setText("")
            return
        self._project_path = config.project_path

        tc = config.train_config
        self.task_combo.setCurrentText(tc.task)
        self.size_combo.setCurrentText(tc.model_size)
        self.epochs_spin.setValue(tc.epochs)
        self.batch_spin.setValue(tc.batch)
        self.imgsz_combo.setCurrentText(str(tc.imgsz))
        self.device_edit.setText(tc.device)
        self.optimizer_combo.setCurrentText(tc.optimizer)
        self.lr_spin.setValue(tc.lr0)
        self.patience_spin.setValue(tc.patience)
        self.workers_spin.setValue(tc.workers)
        self.cache_check.setChecked(tc.cache)
        self.seed_spin.setValue(tc.seed)
        self.plots_check.setChecked(tc.plots)
        self.close_mosaic_spin.setValue(tc.close_mosaic)

        # 恢复模型系列
        family_key = tc.model_family or "yolo26"
        family_display = next((k for k, v in MODEL_FAMILY_MAP.items() if v == family_key), "YOLO26")
        self.family_combo.setCurrentText(family_display)
        self.custom_model_edit.setText(tc.pretrained_model or "")

        # 恢复增强配置
        self.aug_enabled_check.setChecked(tc.augmentation_enabled)
        self._set_aug_preset_key(tc.augmentation_preset)
        self.hsv_h_spin.setValue(tc.hsv_h)
        self.hsv_s_spin.setValue(tc.hsv_s)
        self.hsv_v_spin.setValue(tc.hsv_v)
        self.degrees_spin.setValue(tc.degrees)
        self.translate_spin.setValue(tc.translate)
        self.scale_spin.setValue(tc.scale)
        self.shear_spin.setValue(tc.shear)
        self.perspective_spin.setValue(tc.perspective)
        self.flipud_spin.setValue(tc.flipud)
        self.fliplr_spin.setValue(tc.fliplr)
        self.mosaic_spin.setValue(tc.mosaic)
        self.mixup_spin.setValue(tc.mixup)
        self.cutmix_spin.setValue(tc.cutmix)
        self.copy_paste_spin.setValue(tc.copy_paste)
        self.erasing_spin.setValue(tc.erasing)
        self.auto_augment_combo.setCurrentText(tc.auto_augment or "randaugment")
        self._apply_task_augmentation_advice()

        data_yaml = Path(config.project_path) / "datasets" / "data.yaml"
        self.data_edit.setText(str(data_yaml) if data_yaml.exists() else "")
