from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Union


AUGMENTATION_PRESET_ALIASES = {
    "off": "off",
    "关闭": "off",
    "light": "light",
    "轻度": "light",
    "default": "default",
    "默认": "default",
    "strong": "strong",
    "强增强": "strong",
    "custom": "custom",
    "自定义": "custom",
}


def normalize_augmentation_preset(value: object) -> str:
    preset = str(value or "default").strip()
    return AUGMENTATION_PRESET_ALIASES.get(preset, "default")


@dataclass
class ClassItem:
    name: str = ""
    color: str = "#FF0000"
    kpt_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClassItem:
        return cls(
            name=data.get("name", ""),
            color=data.get("color", "#FF0000"),
            kpt_count=data.get("kpt_count", 0),
        )


@dataclass
class TrainConfig:
    task: str = "detect"
    model_size: str = "n"
    data: str = ""
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = ""
    optimizer: str = "auto"
    lr0: float = 0.01
    patience: int = 100
    project: str = ""
    name: str = ""
    workers: int = 8
    cache: bool = False
    seed: int = 0
    plots: bool = True
    close_mosaic: int = 10
    model_family: str = "yolo26"
    pretrained_model: str = ""
    augmentation_enabled: bool = True
    augmentation_preset: str = "default"
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.4
    auto_augment: str = "randaugment"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["augmentation_preset"] = normalize_augmentation_preset(data.get("augmentation_preset"))
        return data

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        return cls(
            task=data.get("task", "detect"),
            model_size=data.get("model_size", "n"),
            data=data.get("data", ""),
            epochs=data.get("epochs", 100),
            batch=data.get("batch", 16),
            imgsz=data.get("imgsz", 640),
            device=data.get("device", ""),
            optimizer=data.get("optimizer", "auto"),
            lr0=data.get("lr0", 0.01),
            patience=data.get("patience", 100),
            project=data.get("project", ""),
            name=data.get("name", ""),
            workers=data.get("workers", 8),
            cache=data.get("cache", False),
            seed=data.get("seed", 0),
            plots=data.get("plots", True),
            close_mosaic=data.get("close_mosaic", 10),
            model_family=data.get("model_family", "yolo26"),
            pretrained_model=data.get("pretrained_model", ""),
            augmentation_enabled=data.get("augmentation_enabled", True),
            augmentation_preset=normalize_augmentation_preset(data.get("augmentation_preset", "default")),
            hsv_h=data.get("hsv_h", 0.015),
            hsv_s=data.get("hsv_s", 0.7),
            hsv_v=data.get("hsv_v", 0.4),
            degrees=data.get("degrees", 0.0),
            translate=data.get("translate", 0.1),
            scale=data.get("scale", 0.5),
            shear=data.get("shear", 0.0),
            perspective=data.get("perspective", 0.0),
            flipud=data.get("flipud", 0.0),
            fliplr=data.get("fliplr", 0.5),
            mosaic=data.get("mosaic", 1.0),
            mixup=data.get("mixup", 0.0),
            cutmix=data.get("cutmix", 0.0),
            copy_paste=data.get("copy_paste", 0.0),
            erasing=data.get("erasing", 0.4),
            auto_augment=data.get("auto_augment", "randaugment"),
        )


@dataclass
class ProjectConfig:
    project_name: str = ""
    project_path: str = ""
    classes: List[ClassItem] = field(default_factory=list)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    created_at: str = ""
    last_opened: str = ""

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "project_path": self.project_path,
            "classes": [c.to_dict() for c in self.classes],
            "train_config": self.train_config.to_dict(),
            "created_at": self.created_at,
            "last_opened": self.last_opened,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProjectConfig:
        classes = [ClassItem.from_dict(c) for c in data.get("classes", [])]
        train_config = TrainConfig.from_dict(data.get("train_config", {}))
        return cls(
            project_name=data.get("project_name", ""),
            project_path=data.get("project_path", ""),
            classes=classes,
            train_config=train_config,
            created_at=data.get("created_at", ""),
            last_opened=data.get("last_opened", ""),
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        # 原子写入:先写临时文件,再 os.replace 替换
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, str(path))
        except Exception:
            # 异常时清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Union[str, Path]) -> ProjectConfig:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
