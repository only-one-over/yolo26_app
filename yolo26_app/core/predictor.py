from typing import Optional, Tuple

import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOPredictor:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.model_path: str = ""
        self._is_onnx: bool = False
        self._onnx_cpu_fallback: bool = False
        self._shown_onnx_diag: bool = False
        self._model_task: str = ""
        self._onnx_error: str = ""

    def load_model(self, path: str, task: str = "") -> bool:
        try:
            if task and path.lower().endswith(".onnx"):
                self.model = YOLO(path, task=task)
                self._model_task = task
            else:
                self.model = YOLO(path)
                self._model_task = getattr(self.model, "task", "") or ""
            self.model_path = path
            self._is_onnx = path.lower().endswith(".onnx")
            self._onnx_cpu_fallback = False
            self._shown_onnx_diag = False
            self._onnx_error = ""
            if self._is_onnx:
                self._verify_onnx_model()
            return True
        except Exception:
            self.model = None
            self.model_path = ""
            self._is_onnx = False
            self._onnx_cpu_fallback = False
            self._model_task = ""
            return False

    def predict_image(self, image_path: str, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640, device: str = "", max_det: int = 300) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return np.array([]), None
        image = cv2.imread(image_path)
        if image is None:
            return np.array([]), None
        half = self._should_half()
        predict_kwargs = dict(source=image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False, half=half)
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
        half = self._should_half()
        predict_kwargs = dict(source=frame_np, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False, half=half)
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
        exported_path = self.model.export(format=format, **kwargs)
        exported_path = str(exported_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dest = os.path.join(output_dir, os.path.basename(exported_path))
            dest = self._unique_path(dest)
            shutil.move(exported_path, dest)
        else:
            dest = exported_path
        # Verify ONNX models
        if format.lower() == "onnx":
            success, error_msg = self._verify_exported_model(dest)
            return dest, success, error_msg
        return dest, True, ""

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

    def _verify_onnx_model(self) -> None:
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            predict_kwargs = dict(source=dummy, verbose=False)
            if self._model_task:
                predict_kwargs["task"] = self._model_task
            self.model.predict(**predict_kwargs)
        except Exception:
            self._reload_onnx_cpu()

    def _reload_onnx_cpu(self) -> None:
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            kwargs = {}
            if self._model_task:
                kwargs["task"] = self._model_task
            self.model = YOLO(self.model_path, **kwargs)
            self._onnx_cpu_fallback = True
        except Exception as e:
            self._onnx_error = str(e)
        finally:
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

    def _verify_exported_model(self, model_path: str) -> Tuple[bool, str]:
        """Verify exported ONNX model by running a dummy inference.
        
        Returns:
            Tuple of (success, error_message). If success is True, error_message is empty.
        """
        try:
            kwargs = {}
            if self._model_task:
                kwargs["task"] = self._model_task
            test_model = YOLO(model_path, **kwargs)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            test_model.predict(source=dummy, verbose=False)
            return True, ""
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

    @property
    def is_onnx(self) -> bool:
        return self._is_onnx

    def get_onnx_diag(self) -> str:
        if not self._is_onnx:
            return ""
        if self._onnx_cpu_fallback:
            diag = "ONNX 模型已切换为 CPU 推理模式（GPU 推理异常）"
            if hasattr(self, '_onnx_error') and self._onnx_error:
                diag += f"\nCPU 重载失败: {self._onnx_error}"
            return diag
        if not self._shown_onnx_diag:
            self._shown_onnx_diag = True
            return "ONNX 模型推理未检测到目标\n可能原因: onnxruntime-gpu 与 CUDA 版本不匹配\n建议: pip install onnxruntime (使用CPU推理)"
        return ""

    def _should_half(self) -> bool:
        if self._is_onnx:
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

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
