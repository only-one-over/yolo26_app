import logging
import os
import glob as _glob
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from PyQt6.QtCore import QRectF, QPointF
from PyQt6.QtGui import QPolygonF

from yolo26_app.core.annotation_canvas import AnnotationItem

logger = logging.getLogger(__name__)


class YOLOPreAnnotator:
    """使用已加载的 YOLO 模型对图片进行自动预标注"""

    def __init__(self) -> None:
        self._model = None

    def set_model(self, model) -> None:
        self._model = model

    def annotate(self, image_path: str, conf: float = 0.25, class_mapping: Optional[Dict[int, int]] = None) -> List[AnnotationItem]:
        """
        对图片进行 YOLO 推理并返回 AnnotationItem 列表

        Args:
            image_path: 图片路径
            conf: 置信度阈值
            class_mapping: YOLO 类别ID → 项目类别索引的映射，如果为 None 则使用 YOLO 原始类别ID

        Returns:
            AnnotationItem 列表
        """
        if self._model is None:
            return []

        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            h, w = image.shape[:2]

            half = False
            try:
                import torch
                half = torch.cuda.is_available()
            except ImportError:
                pass
            results = self._model.predict(source=image, conf=conf, verbose=False, half=half)
            if not results or len(results) == 0:
                return []

            annotations: List[AnnotationItem] = []
            result = results[0]

            # 处理检测框
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    class_index = class_mapping.get(cls_id, cls_id) if class_mapping else cls_id

                    annotations.append(AnnotationItem(
                        class_index=class_index,
                        rect=QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        item_type="rect",
                    ))

            # 处理分割掩码
            if hasattr(result, "masks") and result.masks is not None:
                masks = result.masks.data
                if masks is not None and len(masks) > 0:
                    for i, mask in enumerate(masks):
                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        mask_resized = cv2.resize(mask_np, (w, h))
                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue
                        largest = max(contours, key=cv2.contourArea)
                        epsilon = 0.005 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, epsilon, True)

                        points: List[QPointF] = []
                        for pt in approx:
                            points.append(QPointF(float(pt[0][0]), float(pt[0][1])))

                        if len(points) >= 3:
                            cls_id = int(result.boxes[i].cls[0]) if result.boxes is not None and i < len(result.boxes) else 0
                            class_index = class_mapping.get(cls_id, cls_id) if class_mapping else cls_id
                            annotations.append(AnnotationItem(
                                class_index=class_index,
                                polygon=QPolygonF(points),
                                item_type="polygon",
                            ))

            return annotations

        except Exception as e:
            logger.error(f"YOLO 预标注失败: {e}")
            return []


SAM2_MODEL_URLS = {
    "sam2.1_hiera_t": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_s": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_b+": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_l": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

GROUNDING_DINO_MODEL_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"


class SAMAnnotator:
    """使用 SAM 2 进行交互式分割标注"""

    _SAM2_CONFIG_MAP = {
        "sam2.1_hiera_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2_hiera_t": "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_s": "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_b+": "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_l": "configs/sam2/sam2_hiera_l.yaml",
    }

    def __init__(self) -> None:
        self._predictor = None
        self._available = False
        self._init_sam()

    def _init_sam(self) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import sam2
            self._sam2_package_dir = os.path.dirname(sam2.__file__)
            self._build_sam2 = build_sam2
            self._predictor_cls = SAM2ImagePredictor
            self._available = True
        except ImportError:
            self._sam2_package_dir = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def resolve_config_path(self, config_path: str) -> str:
        if os.path.isabs(config_path):
            return config_path
        if self._sam2_package_dir is not None:
            abs_path = os.path.join(self._sam2_package_dir, config_path)
            if os.path.isfile(abs_path):
                return abs_path
        return config_path

    @staticmethod
    def scan_model_file(directory: str) -> Optional[Tuple[str, str, str]]:
        for prefix, config_path in SAMAnnotator._SAM2_CONFIG_MAP.items():
            for ext in (".pt", ".pth"):
                pattern = os.path.join(directory, f"{prefix}*{ext}")
                matches = _glob.glob(pattern)
                if matches:
                    return matches[0], prefix, config_path
        return None

    def load_model(self, model_path: str, config_path: str, device: str = "cuda") -> bool:
        if not self._available:
            return False
        try:
            import torch
            resolved_config = self.resolve_config_path(config_path)
            sam = self._build_sam2(resolved_config, model_path, device=device)
            self._predictor = self._predictor_cls(sam)
            return True
        except Exception as e:
            logger.error(f"SAM 2 模型加载失败: {e}")
            return False



class GroundingDINOAnnotator:
    """使用 Grounding DINO 进行文本驱动的零样本检测标注"""

    def __init__(self) -> None:
        self._model = None
        self._available = False
        self._init_dino()

    def _init_dino(self) -> None:
        try:
            from groundingdino.util.inference import load_model, predict
            self._load_model = load_model
            self._predict_fn = predict
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def load_model(self, config_path: str, weights_path: str) -> bool:
        if not self._available:
            return False
        try:
            self._model = self._load_model(config_path, weights_path)
            return True
        except Exception as e:
            logger.error(f"Grounding DINO 模型加载失败: {e}")
            return False

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        class_mapping: Optional[Dict[str, int]] = None,
    ) -> List[AnnotationItem]:
        """
        使用文本描述进行零样本检测

        Args:
            image_path: 图片路径
            text_prompt: 文本描述，如 "car . person ."
            box_threshold: 检测框阈值
            text_threshold: 文本匹配阈值
            class_mapping: 文本标签 → 项目类别索引的映射

        Returns:
            AnnotationItem 列表
        """
        if self._model is None:
            return []

        try:
            import torch

            image = cv2.imread(image_path)
            if image is None:
                return []

            boxes, logits, phrases = self._predict_fn(
                model=self._model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            if boxes is None or len(boxes) == 0:
                return []

            h, w = image.shape[:2]
            annotations: List[AnnotationItem] = []

            for i, (box, phrase) in enumerate(zip(boxes, phrases)):
                cx, cy, bw, bh = box.tolist()
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                bw_px = bw * w
                bh_px = bh * h

                class_index = 0
                if class_mapping and phrase in class_mapping:
                    class_index = class_mapping[phrase]

                annotations.append(AnnotationItem(
                    class_index=class_index,
                    rect=QRectF(float(x1), float(y1), float(bw_px), float(bh_px)),
                    item_type="rect",
                ))

            return annotations

        except Exception as e:
            logger.error(f"Grounding DINO 检测失败: {e}")
            return []
