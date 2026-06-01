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

    def load_model(self, path: str) -> bool:
        try:
            self.model = YOLO(path)
            self.model_path = path
            self._is_onnx = path.lower().endswith(".onnx")
            self._onnx_cpu_fallback = False
            self._shown_onnx_diag = False
            if self._is_onnx:
                self._verify_onnx_model()
            return True
        except Exception:
            self.model = None
            self.model_path = ""
            self._is_onnx = False
            self._onnx_cpu_fallback = False
            return False

    def predict_image(self, image_path: str, conf: float = 0.25) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return np.array([]), None
        image = cv2.imread(image_path)
        if image is None:
            return np.array([]), None
        half = self._should_half()
        results = self.model.predict(source=image, conf=conf, verbose=False, half=half)
        if results and len(results) > 0:
            annotated = self._draw_results(image, results[0])
            return annotated, results[0]
        return image, None

    def predict_frame(self, frame_np: np.ndarray, conf: float = 0.25) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return frame_np, None
        half = self._should_half()
        results = self.model.predict(source=frame_np, conf=conf, verbose=False, half=half)
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
        try:
            metrics = self.model.val(data=data, verbose=False)
            result = {
                "map50": float(metrics.box.map50) if hasattr(metrics, "box") else 0.0,
                "map50_95": float(metrics.box.map) if hasattr(metrics, "box") else 0.0,
            }
            return result
        except Exception:
            return {}

    def export_model(self, format: str, output_dir: str) -> str:
        if self.model is None:
            raise RuntimeError("模型未加载")
        if format.lower() == "onnx":
            exported_path = self.model.export(format=format, simplify=True)
        else:
            exported_path = self.model.export(format=format)
        exported_path = str(exported_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            dest = os.path.join(output_dir, os.path.basename(exported_path))
            shutil.move(exported_path, dest)
        else:
            dest = exported_path
        if format.lower() == "onnx":
            self._verify_exported_model(dest)
        return dest

    def _verify_onnx_model(self) -> None:
        try:
            import torch
            dummy = torch.zeros(1, 3, 640, 640)
            self.model.predict(source=dummy, verbose=False)
        except Exception:
            self._reload_onnx_cpu()

    def _reload_onnx_cpu(self) -> None:
        try:
            old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.model = YOLO(self.model_path)
            self._onnx_cpu_fallback = True
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        except Exception:
            pass

    def _verify_exported_model(self, model_path: str) -> None:
        try:
            test_model = YOLO(model_path)
            import torch
            dummy = torch.zeros(1, 3, 640, 640)
            test_model.predict(source=dummy, verbose=False)
        except Exception:
            pass

    def get_model_info(self) -> dict:
        if self.model is None:
            return {}
        info: dict = {}
        try:
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
            return "ONNX 模型已切换为 CPU 推理模式（GPU 推理异常）"
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
