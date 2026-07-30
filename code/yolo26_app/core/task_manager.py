import uuid
from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QObject, QTimer, QThread, pyqtSignal

from yolo26_app.core.logger import get_logger

logger = get_logger(__name__)


class _TaskWorker(QThread):
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str, str)

    def __init__(self, task_id: str, fn: Callable, parent=None) -> None:
        super().__init__(parent)
        self.task_id = task_id
        self.fn = fn
        self._stop_flag = False

    def request_stop(self) -> None:
        """设置停止标志，由 fn 内部协作检查。"""
        self._stop_flag = True

    @property
    def stop_requested(self) -> bool:
        return self._stop_flag

    def run(self) -> None:
        try:
            result = self.fn()
            if self._stop_flag:
                self.error.emit(self.task_id, "任务已取消")
            else:
                self.finished.emit(self.task_id, result)
        except Exception as e:
            self.error.emit(self.task_id, str(e))


class TaskManager(QObject):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tasks: Dict[str, _TaskWorker] = {}
        self._timers: Dict[str, QTimer] = {}
        self._on_done: Dict[str, Callable] = {}
        self._on_error: Dict[str, Callable] = {}

    def submit(
        self,
        fn: Callable,
        on_done: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        timeout: float = 30.0,
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        worker = _TaskWorker(task_id, fn, self)
        worker.finished.connect(self._handle_finished)
        worker.error.connect(self._handle_error)
        worker.finished.connect(lambda tid=task_id: self._cleanup(tid))
        worker.error.connect(lambda tid=task_id: self._cleanup(tid))
        self._tasks[task_id] = worker
        self._on_done[task_id] = on_done
        self._on_error[task_id] = on_error
        if timeout > 0:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda tid=task_id: self._handle_timeout(tid))
            timer.start(int(timeout * 1000))
            self._timers[task_id] = timer
        worker.start()
        return task_id

    def cancel(self, task_id: str) -> None:
        """请求取消任务：设置停止标志并清理回调，不阻塞等待。"""
        worker = self._tasks.get(task_id)
        if worker is not None:
            worker.request_stop()
        self._cleanup(task_id)

    def shutdown(self) -> None:
        for task_id in list(self._tasks.keys()):
            worker = self._tasks.get(task_id)
            if worker is not None:
                worker.request_stop()
            self._cleanup(task_id)

    def _handle_finished(self, task_id: str, result: Any) -> None:
        callback = self._on_done.pop(task_id, None)
        self._on_error.pop(task_id, None)
        if callback:
            callback(result)

    def _handle_error(self, task_id: str, error_msg: str) -> None:
        callback = self._on_error.pop(task_id, None)
        self._on_done.pop(task_id, None)
        if callback:
            callback(error_msg)

    def _handle_timeout(self, task_id: str) -> None:
        worker = self._tasks.get(task_id)
        if worker is not None:
            worker.request_stop()
        callback = self._on_error.pop(task_id, None)
        self._on_done.pop(task_id, None)
        self._cleanup(task_id)
        if callback:
            callback("任务超时")

    def _cleanup(self, task_id: str) -> None:
        timer = self._timers.pop(task_id, None)
        if timer:
            timer.stop()
        # 不从 _tasks 中移除仍在运行的 worker，由 finished/error 信号的 lambda 清理
        # 但若已经完成（信号已触发），直接移除引用
        worker = self._tasks.get(task_id)
        if worker is not None and not worker.isRunning():
            self._tasks.pop(task_id, None)
