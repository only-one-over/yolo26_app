from typing import Optional, Tuple

import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

from yolo26_app.core.logger import get_logger

_TENSORRT_BUILDER_FLAG_ALIASES = (("FP16", "kFP16"), ("INT8", "kINT8"), ("TF32", "kTF32"))

logger = get_logger(__name__)


class YOLOPredictor:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.model_path: str = ""
        self._is_onnx: bool = False
        self._model_task: str = ""

    def load_model(self, path: str, task: str = "") -> bool:
        if self.model is not None:
            del self.model
            import torch
            torch.cuda.empty_cache()
        try:
            if task and path.lower().endswith((".onnx", ".engine")):
                self.model = YOLO(path, task=task)
                self._model_task = task
            else:
                self.model = YOLO(path)
                self._model_task = getattr(self.model, "task", "") or ""
            self.model_path = path
            self._is_onnx = path.lower().endswith(".onnx")
            return True
        except Exception:
            self.model = None
            self.model_path = ""
            self._is_onnx = False
            self._model_task = ""
            return False

    def predict_image(self, image_path: str, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640, device: str = "", max_det: int = 300) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return np.array([]), None
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return np.array([]), None
        quantize = self._should_quantize()
        predict_kwargs = dict(source=image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)
        if quantize is not None:
            predict_kwargs["quantize"] = quantize
        if device:
            predict_kwargs["device"] = device
        if self._is_onnx and self._model_task:
            predict_kwargs["task"] = self._model_task
        results = self.model.predict(**predict_kwargs)
        if results and len(results) > 0:
            annotated = self._draw_results(image, results[0])
            return annotated, results[0]
        return image, None

    def predict_frame(self, frame_np: np.ndarray, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640, device: str = "", max_det: int = 300) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return frame_np, None
        quantize = self._should_quantize()
        predict_kwargs = dict(source=frame_np, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)
        if quantize is not None:
            predict_kwargs["quantize"] = quantize
        if device:
            predict_kwargs["device"] = device
        if self._is_onnx and self._model_task:
            predict_kwargs["task"] = self._model_task
        results = self.model.predict(**predict_kwargs)
        if results and len(results) > 0:
            annotated = self._draw_results(frame_np, results[0])
            return annotated, results[0]
        return frame_np, None

    def validate_model(self, data: str) -> dict:
        if self.model is None:
            raise RuntimeError("模型未加载")
        if not self.model_path.lower().endswith(".pt"):
            ext = os.path.splitext(self.model_path)[1] if self.model_path else "未知"
            raise RuntimeError(f"{ext} 格式不支持验证，验证功能仅支持 .pt (PyTorch) 模型")
        metrics = self.model.val(data=data, verbose=False)
        # 获取任务类型
        task = self._model_task or getattr(self.model, "task", "detect") or "detect"
        result = {"task": task}
        # 根据任务类型读取对应指标
        if task == "detect":
            if hasattr(metrics, "box"):
                result["map50"] = float(metrics.box.map50) if metrics.box.map50 is not None else 0.0
                result["map50_95"] = float(metrics.box.map) if metrics.box.map is not None else 0.0
        elif task == "segment":
            if hasattr(metrics, "seg"):
                result["map50"] = float(metrics.seg.map50) if metrics.seg.map50 is not None else 0.0
                result["map50_95"] = float(metrics.seg.map) if metrics.seg.map is not None else 0.0
            # 可选：同时显示 box 指标
            if hasattr(metrics, "box"):
                result["box_map50"] = float(metrics.box.map50) if metrics.box.map50 is not None else 0.0
                result["box_map50_95"] = float(metrics.box.map) if metrics.box.map is not None else 0.0
        elif task == "pose":
            if hasattr(metrics, "pose"):
                result["map50"] = float(metrics.pose.map50) if metrics.pose.map50 is not None else 0.0
                result["map50_95"] = float(metrics.pose.map) if metrics.pose.map is not None else 0.0
        elif task == "classify":
            if hasattr(metrics, "top1"):
                result["top1"] = float(metrics.top1) if metrics.top1 is not None else 0.0
            if hasattr(metrics, "top5"):
                result["top5"] = float(metrics.top5) if metrics.top5 is not None else 0.0
        else:
            # 默认使用 box 指标（包括 obb 等其他任务）
            if hasattr(metrics, "box"):
                result["map50"] = float(metrics.box.map50) if metrics.box.map50 is not None else 0.0
                result["map50_95"] = float(metrics.box.map) if metrics.box.map is not None else 0.0
        return result

    def export_model(self, format: str, output_dir: str, **kwargs) -> Tuple[str, bool, str]:
        """Export model to specified format.

        Returns:
            Tuple of (path, success, error_message).
            - path: The exported model path
            - success: True if export and verification both succeed, False otherwise
            - error_message: Empty if success, otherwise contains the verification error
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        export_format = format.lower()
        export_kwargs = self._prepare_export_kwargs(export_format, kwargs)
        if export_format == "engine":
            self._validate_tensorrt_environment(export_kwargs)

        # 预测最终输出路径 final_path
        src_dir = os.path.dirname(self.model_path)
        src_stem = os.path.splitext(os.path.basename(self.model_path))[0]
        predicted_name = src_stem + "." + export_format
        if output_dir:
            final_path = os.path.join(output_dir, predicted_name)
        else:
            final_path = os.path.join(src_dir, predicted_name)

        logger.info("准备导出模型到: %s (格式: %s)", final_path, export_format)

        # 备份现有 final_path(若存在)
        bak_path = final_path + ".bak"
        has_backup = False
        if os.path.exists(final_path):
            try:
                shutil.move(final_path, bak_path)
                has_backup = True
                logger.info("已备份现有导出文件: %s -> %s", final_path, bak_path)
            except Exception as e:
                logger.warning("备份现有导出文件失败: %s", e)
        else:
            logger.info("导出前 %s 不存在,跳过备份", final_path)

        export_success = False
        try:
            try:
                exported_path = self.model.export(format=export_format, **export_kwargs)
            except AttributeError as exc:
                msg = str(exc)
                if "BuilderFlag" in msg or "tensorrt_bindings" in msg:
                    raise RuntimeError(
                        "TensorRT 导出失败：当前 ultralytics 与 TensorRT 版本不兼容"
                        f"（{msg}）。建议升级 ultralytics（pip install -U ultralytics）"
                        "或降级 TensorRT 至 8.x。"
                    ) from exc
                raise

            exported_path = str(exported_path)
            # 移动到 final_path(若不在同一位置)
            if os.path.abspath(exported_path) != os.path.abspath(final_path):
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                shutil.move(exported_path, final_path)
            logger.info("模型已导出到: %s", final_path)

            # 验证(仅对 onnx/engine 做可加载性验证)
            if export_format in {"onnx", "engine"}:
                verify_device = export_kwargs.get("device", "") if export_format == "engine" else ""
                success, error_msg = self._verify_exported_model(
                    final_path,
                    imgsz=export_kwargs.get("imgsz", 640),
                    device=str(verify_device),
                )
                if not success:
                    logger.error("导出文件验证失败: %s", error_msg)
                    self._restore_export_backup(final_path, bak_path, has_backup)
                    return final_path, False, error_msg
                logger.info("导出文件验证通过: %s", final_path)

            export_success = True
            return final_path, True, ""
        except Exception:
            logger.error("模型导出异常,尝试恢复备份", exc_info=True)
            self._restore_export_backup(final_path, bak_path, has_backup)
            raise
        finally:
            # 导出成功后清理 .bak(若仍存在)
            if export_success and has_backup and os.path.exists(bak_path):
                try:
                    os.remove(bak_path)
                    logger.info("已清理备份文件: %s", bak_path)
                except Exception as e:
                    logger.warning("清理备份文件失败: %s", e)

    def _restore_export_backup(self, final_path: str, bak_path: str, has_backup: bool) -> None:
        """恢复备份文件到 final_path(若存在备份)。"""
        if not has_backup:
            return
        try:
            if os.path.exists(final_path):
                if os.path.isdir(final_path):
                    shutil.rmtree(final_path, ignore_errors=True)
                else:
                    os.remove(final_path)
            shutil.move(bak_path, final_path)
            logger.info("已恢复备份: %s -> %s", bak_path, final_path)
        except Exception as e:
            logger.error("恢复备份失败: %s", e)

    def _prepare_export_kwargs(self, format: str, kwargs: dict) -> dict:
        prepared = dict(kwargs)
        is_engine = format == "engine"

        quantize = prepared.pop("quantize", None)
        if quantize not in (None, 8, 16, 32):
            raise ValueError("TensorRT quantize 仅支持 8、16、32 或不设置")
        prepared.pop("half", None)
        prepared.pop("int8", None)

        if quantize == 8 and is_engine:
            data_path = str(prepared.get("data", "")).strip()
            if not data_path:
                raise ValueError("TensorRT INT8 导出必须提供校准数据集 data.yaml")
            if not os.path.isfile(data_path):
                raise FileNotFoundError(f"TensorRT INT8 校准数据不存在: {data_path}")
            fraction = float(prepared.get("fraction", 1.0))
            if not 0.0 < fraction <= 1.0:
                raise ValueError("TensorRT INT8 fraction 必须在 0.0 到 1.0 之间")
            prepared["dynamic"] = True

        if quantize in (8, 16):
            if self._ultralytics_supports_export_arg("quantize"):
                prepared["quantize"] = quantize
            elif quantize == 16:
                prepared["half"] = True
            else:
                prepared["int8"] = True
                if not self._ultralytics_supports_export_arg("fraction"):
                    prepared.pop("fraction", None)

        if is_engine:
            prepared.setdefault("device", "0")
            workspace = prepared.get("workspace")
            if workspace is not None and float(workspace) <= 0:
                prepared.pop("workspace")
        return prepared

    @staticmethod
    def _ultralytics_supports_export_arg(name: str) -> bool:
        try:
            from ultralytics.cfg import get_cfg

            return hasattr(get_cfg(), name)
        except Exception:
            return False

    @staticmethod
    def _apply_tensorrt_enum_compat() -> bool:
        """为 TensorRT 10.x 的 BuilderFlag 补齐旧式枚举别名（FP16/INT8/TF32）。

        旧版 ultralytics 以 trt.BuilderFlag.FP16 方式访问，而 TensorRT 10 改用 kFP16。
        当旧式属性缺失且存在对应 k 前缀新式成员时，补齐同名别名。全程不抛异常。
        """
        try:
            import tensorrt as trt
        except ImportError:
            return False
        try:
            builder_flag = trt.BuilderFlag
            for old_name, new_name in _TENSORRT_BUILDER_FLAG_ALIASES:
                if hasattr(builder_flag, old_name):
                    continue
                if hasattr(builder_flag, new_name):
                    setattr(builder_flag, old_name, getattr(builder_flag, new_name))
            return True
        except Exception:
            return False

    def _validate_tensorrt_environment(self, kwargs: dict) -> None:
        if self.model_path and not self.model_path.lower().endswith(".pt"):
            raise RuntimeError("TensorRT 导出需要以 .pt PyTorch 模型作为源模型")

        device = str(kwargs.get("device", "0")).strip().lower()
        if device in {"", "cpu"}:
            raise RuntimeError("TensorRT 导出不支持 CPU，请选择 NVIDIA CUDA GPU 设备")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TensorRT 导出需要安装支持 CUDA 的 PyTorch") from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用的 NVIDIA CUDA GPU，无法导出 TensorRT Engine。"
                "请检查 NVIDIA 驱动、CUDA 和 PyTorch CUDA 版本。"
            )

        if device.isdigit() and int(device) >= torch.cuda.device_count():
            raise RuntimeError(
                f"TensorRT 导出设备 GPU {device} 不存在，"
                f"当前检测到 {torch.cuda.device_count()} 个 CUDA GPU"
            )

        if not self._apply_tensorrt_enum_compat():
            raise RuntimeError(
                "TensorRT 枚举兼容补丁应用失败，可能是 TensorRT 与 ultralytics 版本不兼容。"
                "建议升级 ultralytics（pip install -U ultralytics）或降级 TensorRT 至 8.x。"
            )

    @staticmethod
    def _unique_path(path: str) -> str:
        """Return a unique file path, adding numeric suffix on collision."""
        if not os.path.exists(path):
            return path
        base, ext = os.path.splitext(path)
        i = 1
        while True:
            candidate = f"{base}_{i}{ext}"
            if not os.path.exists(candidate):
                return candidate
            i += 1

    def _verify_exported_model(
        self,
        model_path: str,
        imgsz: int = 640,
        device: str = "",
    ) -> Tuple[bool, str]:
        """Verify an exported runtime model by running a dummy inference.
        
        Returns:
            Tuple of (success, error_message). If success is True, error_message is empty.
        """
        try:
            kwargs = {}
            if self._model_task:
                kwargs["task"] = self._model_task
            test_model = YOLO(model_path, **kwargs)
            try:
                if isinstance(imgsz, (tuple, list)):
                    height, width = int(imgsz[0]), int(imgsz[1])
                else:
                    height = width = int(imgsz)
                dummy = np.zeros((height, width, 3), dtype=np.uint8)
                predict_kwargs = {"source": dummy, "imgsz": imgsz, "verbose": False}
                if device:
                    predict_kwargs["device"] = device
                test_model.predict(**predict_kwargs)
                return True, ""
            finally:
                del test_model
                import torch
                torch.cuda.empty_cache()
        except Exception as e:
            return False, str(e)

    def get_model_info(self) -> dict:
        if self.model is None:
            return {}
        info: dict = {}
        try:
            if self._is_onnx and self._model_task:
                info["task"] = self._model_task
            else:
                info["task"] = getattr(self.model, "task", "unknown")
        except Exception:
            info["task"] = "unknown"
        try:
            names = getattr(self.model, "names", {})
            info["class_names"] = list(names.values()) if isinstance(names, dict) else []
        except Exception:
            info["class_names"] = []
        return info

    def _should_quantize(self) -> Optional[int]:
        if self._is_onnx:
            return None
        try:
            import torch
            return 16 if torch.cuda.is_available() else None
        except ImportError:
            return None

    def _guess_onnx_task(self, path: str) -> str:
        """Try to infer task type from ONNX model metadata."""
        try:
            import onnx
            model = onnx.load(path, load_external_data=False)
            # Check metadata_props for task
            for prop in model.metadata_props:
                if prop.key == "task":
                    return prop.value
            # Check doc_string
            if model.doc_string:
                for task in ("segment", "classify", "pose", "obb"):
                    if task in model.doc_string.lower():
                        return task
                if "detect" in model.doc_string.lower():
                    return "detect"
            # Check output names for hints
            for output in model.graph.output:
                name = output.name.lower()
                if "mask" in name:
                    return "segment"
                if "keypoint" in name or "kpt" in name:
                    return "pose"
                if "angle" in name or "rot" in name:
                    return "obb"
                if "probs" in name:
                    return "classify"
        except Exception:
            pass
        return ""

    def _draw_results(self, image_np: np.ndarray, results: object) -> np.ndarray:
        try:
            plotted = results.plot()
            if plotted is not None and isinstance(plotted, np.ndarray):
                return plotted
        except Exception:
            pass
        annotated = image_np.copy()
        try:
            boxes = results.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_id} {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception:
            pass
        try:
            if hasattr(results, "masks") and results.masks is not None:
                masks = results.masks.data
                if masks is not None and len(masks) > 0:
                    for mask in masks:
                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        mask_resized = cv2.resize(mask_np, (annotated.shape[1], annotated.shape[0]))
                        colored = np.zeros_like(annotated, dtype=np.uint8)
                        colored[mask_resized > 0] = (0, 0, 255)
                        annotated = cv2.addWeighted(annotated, 1.0, colored, 0.5, 0)
        except Exception:
            pass
        return annotated
