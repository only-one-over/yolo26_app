import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PyQt6.QtCore import QRectF

from yolo26_app.core.annotation_canvas import AnnotationItem
from yolo26_app.core.config import ClassItem


class YOLOExporter:
    @staticmethod
    def export_dataset(
        annotations_dict: Dict[str, List[AnnotationItem]],
        classes: List[ClassItem],
        output_dir: str,
        train_ratio: float = 0.8,
        task: str = "detect",
        flip_idx: Optional[List[int]] = None,
    ) -> Tuple[str, Dict]:
        # Pre-export validation
        def _validate_annotations(
            annotations_dict: Dict[str, List[AnnotationItem]],
            classes: List[ClassItem],
            task: str,
        ) -> None:
            """Validate annotations before export."""
            nc = len(classes)
            if nc == 0:
                raise ValueError("没有有效标注数据可导出")

            allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

            # Determine kpt_shape for pose task
            kpt_count = 0
            if task == "pose":
                for cls in classes:
                    if hasattr(cls, 'kpt_count') and cls.kpt_count > kpt_count:
                        kpt_count = cls.kpt_count
                if kpt_count == 0:
                    for anns in annotations_dict.values():
                        for ann in anns:
                            if ann.item_type == "keypoint" and len(ann.keypoints) > kpt_count:
                                kpt_count = len(ann.keypoints)

            has_valid_annotation = False

            for img_path_str, anns in annotations_dict.items():
                img_path = Path(img_path_str)

                # Validate image extension
                ext = img_path.suffix.lower()
                if ext not in allowed_extensions:
                    raise ValueError(f"图片扩展名不支持: {ext}")

                for ann in anns:
                    # Validate class_index range
                    if ann.class_index < 0 or ann.class_index >= nc:
                        raise ValueError(f"类别索引超出范围: 发现索引 {ann.class_index}，但只有 {nc} 个类别")

                    # Validate keypoints count for pose task
                    if task == "pose" and kpt_count > 0:
                        if ann.item_type in ("rect", "keypoint") and ann.keypoints:
                            actual_kpt = len(ann.keypoints)
                            if actual_kpt != kpt_count:
                                raise ValueError(f"关键点数量不一致: 实例应有 {kpt_count} 个关键点，实际有 {actual_kpt} 个")

                    # Task-specific writability check
                    if task == "detect":
                        if ann.item_type == "rect":
                            w = ann.rect.width()
                            h = ann.rect.height()
                            if w > 0 and h > 0:
                                has_valid_annotation = True
                        elif ann.item_type == "polygon":
                            bbox = ann.polygon.boundingRect()
                            if bbox.width() > 0 and bbox.height() > 0:
                                has_valid_annotation = True
                    elif task == "segment":
                        if ann.item_type == "polygon" and ann.polygon.size() >= 3:
                            has_valid_annotation = True
                    elif task == "pose":
                        if ann.item_type in ("rect", "keypoint") and ann.keypoints:
                            if ann.rect is not None and ann.rect.width() > 0 and ann.rect.height() > 0 and len(ann.keypoints) == kpt_count:
                                has_valid_annotation = True
                    elif task == "classify":
                        if 0 <= ann.class_index < nc:
                            has_valid_annotation = True

            if not has_valid_annotation:
                raise ValueError("没有有效标注数据可导出")

        # Call validation before starting export
        _validate_annotations(annotations_dict, classes, task)

        out = Path(output_dir)

        # classify 任务使用目录结构导出，不使用 YOLO label 格式
        if task == "classify":
            return YOLOExporter._export_classify_dataset(
                annotations_dict, classes, out, train_ratio
            )

        # 其他任务使用标准 YOLO 格式
        if out.exists():
            # 检查目录是否为空
            if any(out.iterdir()):
                # 目录非空，创建带时间戳子目录
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = out / f"dataset_{timestamp}"
        out.mkdir(parents=True, exist_ok=True)

        dirs = {
            "train_img": out / "images" / "train",
            "val_img": out / "images" / "val",
            "train_lbl": out / "labels" / "train",
            "val_lbl": out / "labels" / "val",
        }

        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        skipped_count = 0

        image_paths = list(annotations_dict.keys())
        random.shuffle(image_paths)
        split_idx = max(1, int(len(image_paths) * train_ratio))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        if not val_paths and len(train_paths) >= 2:
            val_paths = [train_paths.pop()]

        def _process(paths: List[str], img_dir: Path, lbl_dir: Path) -> int:
            nonlocal skipped_count
            processed = 0
            for img_path_str in paths:
                img_path = Path(img_path_str)
                if not img_path.exists():
                    continue

                anns = annotations_dict[img_path_str]
                if not anns:
                    skipped_count += 1
                    continue

                dest_img = img_dir / img_path.name
                shutil.copy2(str(img_path), str(dest_img))

                img = cv2.imread(str(img_path))
                if img is None:
                    skipped_count += 1
                    continue
                img_h, img_w = img.shape[:2]
                if img_w <= 0 or img_h <= 0:
                    skipped_count += 1
                    continue

                label_name = img_path.stem + ".txt"
                label_path = lbl_dir / label_name

                lines: List[str] = []
                for ann in anns:
                    if ann.item_type == "rect":
                        if task == "segment":
                            continue
                        w = ann.rect.width()
                        h = ann.rect.height()
                        if w < 1 or h < 1:
                            continue
                        cx = (ann.rect.x() + w / 2) / img_w
                        cy = (ann.rect.y() + h / 2) / img_h
                        nw = w / img_w
                        nh = h / img_h
                        cx = max(0.0, min(1.0, cx))
                        cy = max(0.0, min(1.0, cy))
                        nw = max(0.0, min(1.0, nw))
                        nh = max(0.0, min(1.0, nh))
                        if nw <= 0 or nh <= 0:
                            continue
                        if task == "pose" and not ann.keypoints:
                            continue
                        if task == "pose" and ann.keypoints:
                            kpt_parts = []
                            for kp in ann.keypoints:
                                kx = kp.x() / img_w
                                ky = kp.y() / img_h
                                kpt_parts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
                            line = f"{ann.class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                            if kpt_parts:
                                line += " " + " ".join(kpt_parts)
                            lines.append(line)
                        else:
                            lines.append(f"{ann.class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    elif ann.item_type == "polygon":
                        if ann.polygon.size() < 3:
                            continue
                        if task == "segment":
                            pts = [(pt.x(), pt.y()) for pt in ann.polygon]
                            pts_np = np.array(pts, dtype=np.float32)
                            peri = cv2.arcLength(pts_np, True)
                            epsilon = 0.005 * peri
                            approx_np = cv2.approxPolyDP(pts_np, epsilon, True)
                            if len(approx_np) < 3:
                                approx_np = pts_np.reshape(-1, 1, 2)
                            coords: List[str] = [str(ann.class_index)]
                            for pt in approx_np.reshape(-1, 2):
                                coords.append(f"{max(0.0, min(1.0, pt[0] / img_w)):.6f}")
                                coords.append(f"{max(0.0, min(1.0, pt[1] / img_h)):.6f}")
                            lines.append(" ".join(coords))
                        else:
                            bbox = ann.polygon.boundingRect()
                            w = bbox.width()
                            h = bbox.height()
                            if w < 1 or h < 1:
                                continue
                            cx = (bbox.x() + w / 2) / img_w
                            cy = (bbox.y() + h / 2) / img_h
                            nw = w / img_w
                            nh = h / img_h
                            cx = max(0.0, min(1.0, cx))
                            cy = max(0.0, min(1.0, cy))
                            nw = max(0.0, min(1.0, nw))
                            nh = max(0.0, min(1.0, nh))
                            if nw <= 0 or nh <= 0:
                                continue
                            lines.append(f"{ann.class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    elif ann.item_type == "keypoint":
                        if task == "pose":
                            if ann.rect is not None and ann.rect.width() > 0 and ann.rect.height() > 0:
                                cx = (ann.rect.x() + ann.rect.width() / 2) / img_w
                                cy = (ann.rect.y() + ann.rect.height() / 2) / img_h
                                nw = ann.rect.width() / img_w
                                nh = ann.rect.height() / img_h
                                cx = max(0.0, min(1.0, cx))
                                cy = max(0.0, min(1.0, cy))
                                nw = max(0.0, min(1.0, nw))
                                nh = max(0.0, min(1.0, nh))
                                if nw <= 0 or nh <= 0:
                                    continue
                                kpt_parts = []
                                for kp in ann.keypoints:
                                    kx = kp.x() / img_w
                                    ky = kp.y() / img_h
                                    kpt_parts.extend([f"{kx:.6f}", f"{ky:.6f}", "2"])
                                line = f"{ann.class_index} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                                if kpt_parts:
                                    line += " " + " ".join(kpt_parts)
                                lines.append(line)

                if not lines:
                    skipped_count += 1
                    continue

                label_path.write_text("\n".join(lines), encoding="utf-8")
                processed += 1
            return processed

        train_count = _process(train_paths, dirs["train_img"], dirs["train_lbl"])
        val_count = _process(val_paths, dirs["val_img"], dirs["val_lbl"])

        if train_count == 0:
            raise ValueError("训练集为空，无法生成有效数据集")
        if val_count == 0:
            raise ValueError("验证集为空，无法生成有效数据集")

        yaml_content = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": {i: c.name for i, c in enumerate(classes)},
        }

        if task == "pose":
            max_kpt = 0
            for cls in classes:
                if hasattr(cls, 'kpt_count') and cls.kpt_count > max_kpt:
                    max_kpt = cls.kpt_count
            if max_kpt == 0:
                for anns in annotations_dict.values():
                    for ann in anns:
                        if ann.item_type == "keypoint" and len(ann.keypoints) > max_kpt:
                            max_kpt = len(ann.keypoints)
            if max_kpt > 0:
                yaml_content["kpt_shape"] = [max_kpt, 3]
                # flip_idx 不自动生成，需用户根据关键点语义配置左右翻转映射
                if flip_idx is not None:
                    yaml_content["flip_idx"] = flip_idx

        yaml_path = out / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            # pose 任务未配置 flip_idx 时添加提示注释
            if task == "pose" and flip_idx is None and "kpt_shape" in yaml_content:
                f.write("# flip_idx: 需用户根据关键点语义配置左右翻转映射\n")

        return str(yaml_path), {"train_count": train_count, "val_count": val_count, "skipped_count": skipped_count}

    @staticmethod
    def _export_classify_dataset(
        annotations_dict: Dict[str, List[AnnotationItem]],
        classes: List[ClassItem],
        out: Path,
        train_ratio: float = 0.8,
    ) -> Tuple[str, Dict]:
        """
        classify 任务使用目录结构导出：
        - train/<class_name>/<image_files>
        - val/<class_name>/<image_files>
        不生成 labels 目录和 .txt 文件
        """
        if out.exists():
            if any(out.iterdir()):
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = out / f"dataset_{timestamp}"
        out.mkdir(parents=True, exist_ok=True)

        # 创建 train 和 val 目录
        train_dir = out / "train"
        val_dir = out / "val"

        # 为每个类别创建子目录
        class_names = [c.name for c in classes]
        for cls_name in class_names:
            (train_dir / cls_name).mkdir(parents=True, exist_ok=True)
            (val_dir / cls_name).mkdir(parents=True, exist_ok=True)

        skipped_count = 0
        train_count = 0
        val_count = 0

        # 分割图片为训练集和验证集
        image_paths = list(annotations_dict.keys())
        random.shuffle(image_paths)
        split_idx = max(1, int(len(image_paths) * train_ratio))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        if not val_paths and len(train_paths) >= 2:
            val_paths = [train_paths.pop()]

        def _get_primary_class_index(anns: List[AnnotationItem]) -> int:
            """获取图片的主要类别（标注数量最多的类别）"""
            class_counts: Dict[int, int] = {}
            for ann in anns:
                class_counts[ann.class_index] = class_counts.get(ann.class_index, 0) + 1
            if not class_counts:
                return -1
            # 返回标注数量最多的类别索引
            return max(class_counts, key=class_counts.get)

        def _process_classify(paths: List[str], dest_dir: Path) -> int:
            nonlocal skipped_count
            processed = 0
            for img_path_str in paths:
                img_path = Path(img_path_str)
                if not img_path.exists():
                    skipped_count += 1
                    continue

                anns = annotations_dict[img_path_str]
                if not anns:
                    skipped_count += 1
                    continue

                # 获取主要类别索引
                primary_class_idx = _get_primary_class_index(anns)
                if primary_class_idx < 0 or primary_class_idx >= len(classes):
                    skipped_count += 1
                    continue

                # 获取类别名称
                class_name = classes[primary_class_idx].name

                # 复制图片到对应类别目录
                dest_img = dest_dir / class_name / img_path.name
                shutil.copy2(str(img_path), str(dest_img))
                processed += 1

            return processed

        train_count = _process_classify(train_paths, train_dir)
        val_count = _process_classify(val_paths, val_dir)

        if train_count == 0:
            raise ValueError("训练集为空，无法生成有效数据集")
        if val_count == 0:
            raise ValueError("验证集为空，无法生成有效数据集")

        # 写入 data.yaml
        yaml_content = {
            "path": str(out.resolve()),
            "train": "train",
            "val": "val",
            "nc": len(classes),
            "names": {i: c.name for i, c in enumerate(classes)},
        }

        yaml_path = out / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        return str(yaml_path), {"train_count": train_count, "val_count": val_count, "skipped_count": skipped_count}
