from threading import RLock
from typing import Any, Optional


class ModelSession:
    """Shared, serialized access to one loaded YOLO model."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.model: Optional[Any] = None
        self.task = ""
        self.device = ""
        self.state = "idle"

    def set_model(self, model: Optional[Any], task: str = "") -> None:
        with self.lock:
            self.model = model
            self.task = task

    def set_state(self, state: str, device: Optional[str] = None) -> None:
        with self.lock:
            self.state = state
            if device is not None:
                self.device = device
