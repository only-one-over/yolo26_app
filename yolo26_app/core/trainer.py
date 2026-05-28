import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from yolo26_app.core.config import TrainConfig


class YOLOTrainer(QThread):
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    TASK_MODEL_MAP = {
        "detect": "yolo26{size}.pt",
        "segment": "yolo26{size}-seg.pt",
        "classify": "yolo26{size}-cls.pt",
        "pose": "yolo26{size}-pose.pt",
    }

    def __init__(self, config: TrainConfig, project_path: str) -> None:
        super().__init__()
        self.config = config
        self.project_path = project_path
        self._stop_flag = False

    def run(self) -> None:
        try:
            from ultralytics import YOLO

            task = self.config.task
            size = self.config.model_size
            model_template = self.TASK_MODEL_MAP.get(task, "yolo26{size}.pt")
            model_file = model_template.format(size=size)

            self.log_signal.emit(f"加载模型: {model_file}")
            model = YOLO(model_file)

            handler = _QtLogHandler(self.log_signal)
            handler.setLevel(logging.INFO)
            logger = logging.getLogger("ultralytics")
            logger.addHandler(handler)

            project_dir = str(Path(self.project_path) / "runs")
            name = self.config.name or "train"

            device = self.config.device if self.config.device else None

            self.log_signal.emit(
                f"开始训练: task={task}, epochs={self.config.epochs}, "
                f"batch={self.config.batch}, imgsz={self.config.imgsz}"
            )

            results = model.train(
                data=self.config.data,
                epochs=self.config.epochs,
                batch=self.config.batch,
                imgsz=self.config.imgsz,
                device=device,
                optimizer=self.config.optimizer,
                lr0=self.config.lr0,
                patience=self.config.patience,
                project=project_dir,
                name=name,
            )

            logger.removeHandler(handler)

            if self._stop_flag:
                self.log_signal.emit("训练已被用户停止")
                self.finished_signal.emit("训练已被用户停止")
                return

            best_path = Path(project_dir) / name / "weights" / "best.pt"
            metrics_parts: list[str] = []

            if results is not None:
                try:
                    metrics_dict = results.results_dict if hasattr(results, "results_dict") else {}
                    for key, val in metrics_dict.items():
                        if isinstance(val, float):
                            metrics_parts.append(f"{key}: {val:.4f}")
                        else:
                            metrics_parts.append(f"{key}: {val}")
                except Exception:
                    pass

            msg = f"训练完成!\n最佳模型: {best_path}"
            if metrics_parts:
                msg += "\n\n指标:\n" + "\n".join(metrics_parts)

            self.finished_signal.emit(msg)

        except Exception as e:
            self.error_signal.emit(f"训练出错: {str(e)}")

    def stop(self) -> None:
        self._stop_flag = True


class _QtLogHandler(logging.Handler):
    def __init__(self, signal: object) -> None:
        super().__init__()
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._signal.emit(msg)
