import json
import multiprocessing
import time
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

CACHE_DIR = Path.home() / ".yolo26_app"
CACHE_FILE = CACHE_DIR / "gpu_cache.json"
STATE_FILE = CACHE_DIR / "app_state.json"
CACHE_TTL = 1800


def _cuda_check_worker(result_queue: multiprocessing.Queue) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            result_queue.put(("gpu", name))
        else:
            result_queue.put(("cpu", ""))
    except Exception as e:
        result_queue.put(("error", str(e)))


def detect_gpu_subprocess(timeout: float = 5.0) -> Tuple[str, str]:
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_cuda_check_worker, args=(result_queue,))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        return ("timeout", "")
    if not result_queue.empty():
        return result_queue.get()
    return ("cpu", "")


def load_gpu_cache() -> Optional[Tuple[str, str]]:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    ts = data.get("timestamp", 0)
    if time.time() - ts > CACHE_TTL:
        return None
    status = data.get("status", "cpu")
    name = data.get("device_name", "")
    return (status, name)


def save_gpu_cache(status: str, device_name: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "status": status,
            "device_name": device_name,
            "timestamp": time.time(),
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except (PermissionError, OSError):
        pass


def load_exit_flag() -> Optional[bool]:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_exit_normal", None)
    except (json.JSONDecodeError, OSError):
        return None


def save_exit_flag(normal: bool) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data = {}
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data["last_exit_normal"] = normal
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (PermissionError, OSError):
        pass


class GPUDetectWorker(QThread):
    result_ready = pyqtSignal(str, str)

    def __init__(self, parent=None, timeout: float = 5.0) -> None:
        super().__init__(parent)
        self.timeout = timeout

    def run(self) -> None:
        cached = load_gpu_cache()
        if cached is not None:
            self.result_ready.emit(cached[0], cached[1])
            return
        status, name = detect_gpu_subprocess(timeout=self.timeout)
        if status == "gpu":
            save_gpu_cache("gpu", name)
        elif status == "cpu":
            save_gpu_cache("cpu", "")
        self.result_ready.emit(status, name)
