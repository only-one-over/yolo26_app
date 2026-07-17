from typing import List, Optional

from yolo26_app.core.config import ClassItem, ProjectConfig

COLOR_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F0B27A", "#82E0AA", "#F1948A", "#85929E", "#73C6B6",
    "#E59866", "#A569BD", "#5DADE2", "#58D68D", "#EB984E",
]


class LabelManager:
    def __init__(self) -> None:
        self._classes: List[ClassItem] = []

    def load_from_project(self, config: ProjectConfig) -> None:
        self._classes = list(config.classes)

    def save_to_project(self, config: ProjectConfig) -> None:
        config.classes = list(self._classes)

    def add_class(self, name: str, kpt_count: int = 0) -> ClassItem:
        color = COLOR_PALETTE[len(self._classes) % len(COLOR_PALETTE)]
        item = ClassItem(name=name, color=color, kpt_count=kpt_count)
        self._classes.append(item)
        return item

    def remove_class(self, index: int) -> None:
        if 0 <= index < len(self._classes):
            self._classes.pop(index)

    def update_class(self, index: int, name: str) -> None:
        if 0 <= index < len(self._classes):
            self._classes[index].name = name

    def get_class_by_index(self, index: int) -> Optional[ClassItem]:
        if 0 <= index < len(self._classes):
            return self._classes[index]
        return None

    def get_class_index(self, name: str) -> int:
        for i, c in enumerate(self._classes):
            if c.name == name:
                return i
        return -1

    def get_all_classes(self) -> List[ClassItem]:
        return list(self._classes)
