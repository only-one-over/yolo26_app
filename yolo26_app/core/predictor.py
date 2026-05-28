from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOPredictor:
    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.model_path: str = ""

    def load_model(self, path: str) -> bool:
        try:
            self.model = YOLO(path)
            self.model_path = path
            return True
        except Exception:
            self.model = None
            self.model_path = ""
            return False

    def predict_image(self, image_path: str, conf: float = 0.25) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return np.array([]), None
        image = cv2.imread(image_path)
        if image is None:
            return np.array([]), None
        results = self.model.predict(source=image, conf=conf, verbose=False)
        if results and len(results) > 0:
            annotated = self._draw_results(image, results[0])
            return annotated, results[0]
        return image, None

    def predict_frame(self, frame_np: np.ndarray, conf: float = 0.25) -> Tuple[np.ndarray, object]:
        if self.model is None:
            return frame_np, None
        results = self.model.predict(source=frame_np, conf=conf, verbose=False)
        if results and len(results) > 0:
            annotated = self._draw_results(frame_np, results[0])
            return annotated, results[0]
        return frame_np, None

    def validate_model(self, data: str) -> dict:
        if self.model is None:
            return {}
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
            return ""
        try:
            exported_path = self.model.export(format=format)
            return str(exported_path)
        except Exception:
            return ""

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
