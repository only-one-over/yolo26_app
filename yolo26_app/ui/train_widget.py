import glob
from pathlib import Path
from typing import List, Optional

import yaml

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
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from yolo26_app.core.config import TrainConfig, ProjectConfig
from yolo26_app.core.trainer import YOLOTrainer

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


class TrainWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._trainer: Optional[YOLOTrainer] = None
        self._project_path: str = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        config_group = QGroupBox("训练配置")
        form = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "classify", "pose"])
        form.addRow("任务类型:", self.task_combo)

        self._task_info_label = QLabel(TASK_INFO.get("detect", ""))
        self._task_info_label.setStyleSheet("color: #666; font-size: 11px;")
        self._task_info_label.setWordWrap(True)
        form.addRow("", self._task_info_label)

        self.size_combo = QComboBox()
        self.size_combo.addItems(["n", "s", "m", "l", "x"])
        form.addRow("模型大小:", self.size_combo)

        self._model_info_label = QLabel(MODEL_INFO.get("n", ""))
        self._model_info_label.setStyleSheet("color: #666; font-size: 11px;")
        self._model_info_label.setWordWrap(True)
        form.addRow("", self._model_info_label)

        data_row = QHBoxLayout()
        self.data_edit = QLineEdit()
        self.data_edit.setPlaceholderText("选择数据集 .yaml 文件")
        data_browse = QPushButton("浏览")
        data_browse.setFixedWidth(60)
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
        self.optimizer_combo.addItems(["auto", "SGD", "Adam", "AdamW", "MuSGD"])
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

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(config_group)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        self.size_combo.currentTextChanged.connect(self._update_model_info)
        self.task_combo.currentTextChanged.connect(self._update_task_info)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        layout.addWidget(self.log_text)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #a5d6a7; color: #e0e0e0; }"
        )
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #d32f2f; }"
            "QPushButton:disabled { background-color: #ef9a9a; color: #e0e0e0; }"
        )
        self.stop_btn.clicked.connect(self._on_stop)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.results_group = QGroupBox("训练结果")
        results_layout = QVBoxLayout()
        self.result_model_label = QLabel("最佳模型: -")
        self.result_metrics_label = QLabel("指标: -")
        results_layout.addWidget(self.result_model_label)
        results_layout.addWidget(self.result_metrics_label)
        self.results_group.setLayout(results_layout)
        self.results_group.hide()
        layout.addWidget(self.results_group)

    def _update_model_info(self, size: str) -> None:
        self._model_info_label.setText(MODEL_INFO.get(size, ""))

    def _update_task_info(self, task: str) -> None:
        self._task_info_label.setText(TASK_INFO.get(task, ""))

    def _browse_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置文件", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.data_edit.setText(path)

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
        base = Path(cfg.get("path", ""))
        for key in ("train", "val"):
            rel = cfg.get(key, "")
            img_dir = base / rel if rel else base / "images" / key
            if not img_dir.exists():
                QMessageBox.warning(self, "验证失败", f"目录不存在: {img_dir}")
                return False
            images = glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")) + glob.glob(str(img_dir / "*.bmp"))
            if not images:
                QMessageBox.warning(self, "验证失败", f"目录中没有图片: {img_dir}")
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

        self._trainer.start()

    def _on_stop(self) -> None:
        if self._trainer and self._trainer.isRunning():
            self._trainer.stop()
            self.status_label.setText("正在停止训练...")
            self.stop_btn.setEnabled(False)

    def _on_progress(self, current: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"训练中: Epoch {current}/{total}")

    def _on_log(self, message: str) -> None:
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_finished(self, message: str) -> None:
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

    def _on_error(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_form_enabled(True)
        self.status_label.setText("训练出错")
        QMessageBox.critical(self, "训练错误", message)

    def set_project_config(self, config: ProjectConfig) -> None:
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

        data_yaml = Path(config.project_path) / "datasets" / "data.yaml"
        if data_yaml.exists():
            self.data_edit.setText(str(data_yaml))
