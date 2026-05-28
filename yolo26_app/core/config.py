from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Union


@dataclass
class ClassItem:
    name: str = ""
    color: str = "#FF0000"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClassItem:
        return cls(
            name=data.get("name", ""),
            color=data.get("color", "#FF0000"),
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

    def to_dict(self) -> dict:
        return asdict(self)

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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ProjectConfig:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
