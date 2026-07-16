import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from yolo26_app.core.config import TrainConfig
from yolo26_app.core.model_registry import MODEL_FAMILY_TASK_MODEL_MAP, AUGMENTATION_PRESET_LABELS


class YOLOTrainer(QThread):
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, config: TrainConfig, project_path: str) -> None:
        super().__init__()
        self.config = config
        self.project_path = project_path
        self._stop_flag = False

    def _on_train_epoch_end(self, trainer) -> None:
        if self._stop_flag:
            trainer.stop_training = True
            return
        epoch = getattr(trainer, "epoch", 0) + 1
        total = getattr(trainer, "epochs", self.config.epochs)
        self.progress_signal.emit(epoch, total)

    def _on_train_batch_end(self, trainer) -> None:
        """每个 batch 结束时检查停止标志,减少长 epoch 的停止延迟。"""
        if self._stop_flag:
            trainer.stop_training = True
            return

    def _build_augmentation_kwargs(self) -> dict:
        if self.config.augmentation_enabled:
            return {
                "hsv_h": self.config.hsv_h,
                "hsv_s": self.config.hsv_s,
                "hsv_v": self.config.hsv_v,
                "degrees": self.config.degrees,
                "translate": self.config.translate,
                "scale": self.config.scale,
                "shear": self.config.shear,
                "perspective": self.config.perspective,
                "flipud": self.config.flipud,
                "fliplr": self.config.fliplr,
                "mosaic": self.config.mosaic,
                "mixup": self.config.mixup,
                "cutmix": self.config.cutmix,
                "copy_paste": self.config.copy_paste,
                "erasing": self.config.erasing,
                "auto_augment": self.config.auto_augment if self.config.auto_augment else None,
            }

        return {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
        }

    def _augmentation_log_summary(self, aug_kwargs: dict) -> str:
        preset = AUGMENTATION_PRESET_LABELS.get(self.config.augmentation_preset, self.config.augmentation_preset)
        enabled = "启用" if self.config.augmentation_enabled else "关闭"
        status = f"数据增强: {enabled}, 预设={preset}"
        values = (
            f"mosaic={aug_kwargs['mosaic']}, mixup={aug_kwargs['mixup']}, "
            f"cutmix={aug_kwargs['cutmix']}, copy_paste={aug_kwargs['copy_paste']}, "
            f"hsv=({aug_kwargs['hsv_h']}, {aug_kwargs['hsv_s']}, {aug_kwargs['hsv_v']}), "
            f"scale={aug_kwargs['scale']}, fliplr={aug_kwargs['fliplr']}, "
            f"close_mosaic={self.config.close_mosaic}"
        )
        if not self.config.augmentation_enabled:
            return f"{status}; 已显式关闭 Ultralytics 增强参数; {values}"
        return f"{status}; {values}"

    def run(self) -> None:
        model = None
        handler = None
        logger = None
        try:
            from ultralytics import YOLO

            task = self.config.task
            size = self.config.model_size

            if self.config.pretrained_model:
                model_file = self.config.pretrained_model
            else:
                family = self.config.model_family or "yolo26"
                family_map = MODEL_FAMILY_TASK_MODEL_MAP.get(family, MODEL_FAMILY_TASK_MODEL_MAP["yolo26"])
                model_template = family_map.get(task, family_map["detect"])
                model_file = model_template.format(size=size)
                # 预训练模型统一存到 system_model/yolo/，不存在时 ultralytics 自动下载
                from yolo26_app.core.paths import SYSTEM_MODEL_SUBDIRS
                yolo_model_dir = SYSTEM_MODEL_SUBDIRS["yolo"]
                yolo_model_dir.mkdir(parents=True, exist_ok=True)
                model_file = str(yolo_model_dir / model_file)

            self.log_signal.emit(f"加载模型: {model_file}")
            model = YOLO(model_file)

            handler = _QtLogHandler(self.log_signal)
            handler.setLevel(logging.INFO)
            logger = logging.getLogger("ultralytics")
            logger.addHandler(handler)

            model.add_callback("on_train_epoch_end", self._on_train_epoch_end)
            model.add_callback("on_train_batch_end", self._on_train_batch_end)

            project_dir = str(Path(self.project_path) / "runs")
            name = self.config.name or "train"

            device = self.config.device if self.config.device else None

            self.log_signal.emit(
                f"开始训练: task={task}, epochs={self.config.epochs}, "
                f"batch={self.config.batch}, imgsz={self.config.imgsz}"
            )

            aug_kwargs = self._build_augmentation_kwargs()
            self.log_signal.emit(self._augmentation_log_summary(aug_kwargs))

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
                workers=self.config.workers,
                cache=self.config.cache,
                seed=self.config.seed,
                plots=self.config.plots,
                close_mosaic=self.config.close_mosaic,
                **aug_kwargs,
            )

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
        finally:
            if handler is not None and logger is not None:
                logger.removeHandler(handler)
            if model is not None:
                del model
                import torch

                torch.cuda.empty_cache()

    def stop(self) -> None:
        self._stop_flag = True


class _QtLogHandler(logging.Handler):
    def __init__(self, signal: object) -> None:
        super().__init__()
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._signal.emit(msg)
