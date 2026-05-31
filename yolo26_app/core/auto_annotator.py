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

            results = self._model.predict(source=image, conf=conf, verbose=False)
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
            self._build_sam2 = build_sam2
            self._predictor_cls = SAM2ImagePredictor
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

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
            sam = self._build_sam2(config_path, model_path, device=device)
            self._predictor = self._predictor_cls(sam)
            return True
        except Exception as e:
            logger.error(f"SAM 2 模型加载失败: {e}")
            return False


class VideoTracker:
    """使用 OpenCV CSRT 追踪器进行视频帧间标注传播"""

    @staticmethod
    def get_tracker_name() -> str:
        for name, factory in [
            ("CSRT", lambda: cv2.TrackerCSRT_create()),
            ("CSRT", lambda: cv2.legacy.TrackerCSRT_create()),
            ("KCF", lambda: cv2.TrackerKCF_create()),
            ("KCF", lambda: cv2.legacy.TrackerKCF_create()),
            ("MIL", lambda: cv2.TrackerMIL_create()),
        ]:
            try:
                factory()
                return name
            except AttributeError:
                continue
        return "N/A"

    @staticmethod
    def _create_tracker():
        for factory in [
            lambda: cv2.TrackerCSRT_create(),
            lambda: cv2.legacy.TrackerCSRT_create(),
            lambda: cv2.TrackerKCF_create(),
            lambda: cv2.legacy.TrackerKCF_create(),
            lambda: cv2.TrackerMIL_create(),
        ]:
            try:
                return factory()
            except AttributeError:
                continue
        raise RuntimeError(
            "OpenCV 追踪器不可用。请安装: pip install opencv-contrib-python"
        )

    def track_frames(
        self,
        image_paths: List[str],
        initial_annotations: List[AnnotationItem],
        max_frames: int = 0,
    ) -> Dict[int, List[AnnotationItem]]:
        """
        从第一帧的矩形标注开始追踪后续帧

        Args:
            image_paths: 图片路径列表（按帧顺序）
            initial_annotations: 第一帧的矩形标注
            max_frames: 最大追踪帧数，0 表示全部

        Returns:
            {帧索引: [AnnotationItem]} 字典
        """
        if not image_paths or not initial_annotations:
            return {}

        result: Dict[int, List[AnnotationItem]] = {0: initial_annotations}
        rect_anns = [a for a in initial_annotations if a.item_type == "rect"]
        if not rect_anns:
            return result

        first_image = cv2.imread(image_paths[0])
        if first_image is None:
            return result

        trackers = []
        img_h, img_w = first_image.shape[:2]
        for ann in rect_anns:
            x = max(0, int(ann.rect.x()))
            y = max(0, int(ann.rect.y()))
            w = max(1, int(ann.rect.width()))
            h = max(1, int(ann.rect.height()))
            if x >= img_w or y >= img_h:
                continue
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            if w < 2 or h < 2:
                continue
            tracker = self._create_tracker()
            try:
                tracker.init(first_image, (x, y, w, h))
            except cv2.error:
                continue
            trackers.append((tracker, ann.class_index))

        end = len(image_paths) if max_frames == 0 else min(len(image_paths), max_frames + 1)

        for frame_idx in range(1, end):
            image = cv2.imread(image_paths[frame_idx])
            if image is None:
                break

            frame_anns: List[AnnotationItem] = []
            all_ok = True
            for tracker, class_index in trackers:
                ok, bbox = tracker.update(image)
                if ok:
                    x, y, w, h = bbox
                    frame_anns.append(AnnotationItem(
                        class_index=class_index,
                        rect=QRectF(float(x), float(y), float(w), float(h)),
                        item_type="rect",
                    ))
                else:
                    all_ok = False

            if not all_ok and not frame_anns:
                break

            result[frame_idx] = frame_anns

        return result


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
