"""Local, append-only timing events for application startup and project loading."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class StartupMetrics:
    """Record relative durations without collecting user media paths or contents."""

    def __init__(self, workspace_root: Path) -> None:
        self._started_at = time.perf_counter()
        self._events_path = workspace_root / "logs" / "startup_metrics.jsonl"

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._started_at) * 1000, 2)

    def mark(self, stage: str, **details: Any) -> None:
        event = {
            "timestamp": time.time(),
            "elapsed_ms": self.elapsed_ms,
            "stage": stage,
            "runtime": "frozen" if getattr(sys, "frozen", False) else "source",
            **details,
        }
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            # Timing must never prevent the GUI from starting.
            pass
