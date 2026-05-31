import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
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
    ) -> Tuple[str, Dict]:
        out = Path(output_dir)
        dirs = {
            "train_img": out / "images" / "train",
            "val_img": out / "images" / "val",
            "train_lbl": out / "labels" / "train",
            "val_lbl": out / "labels" / "val",
        }

        if out.exists():
            shutil.rmtree(out)
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

                if not lines:
                    skipped_count += 1
                    continue

                label_path.write_text("\n".join(lines), encoding="utf-8")
                processed += 1
            return processed

        train_count = _process(train_paths, dirs["train_img"], dirs["train_lbl"])
        val_count = _process(val_paths, dirs["val_img"], dirs["val_lbl"])

        yaml_content = {
            "path": str(out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": {i: c.name for i, c in enumerate(classes)},
        }

        yaml_path = out / "data.yaml"
        lines: List[str] = []
        lines.append(f"path: {yaml_content['path']}")
        lines.append(f"train: {yaml_content['train']}")
        lines.append(f"val: {yaml_content['val']}")
        lines.append(f"nc: {yaml_content['nc']}")
        names_str = ", ".join(f"'{c.name}'" for c in classes)
        lines.append(f"names: [{names_str}]")
        yaml_path.write_text("\n".join(lines), encoding="utf-8")

        return str(yaml_path), {"train_count": train_count, "val_count": val_count, "skipped_count": skipped_count}
