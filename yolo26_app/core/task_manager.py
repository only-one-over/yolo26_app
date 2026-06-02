import uuid
from typing import Any, Callable, Dict, Optional

from PyQt6.QtCore import QObject, QTimer, QThread, pyqtSignal


class _TaskWorker(QThread):
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str, str)

    def __init__(self, task_id: str, fn: Callable, parent=None) -> None:
        super().__init__(parent)
        self.task_id = task_id
        self.fn = fn

    def run(self) -> None:
        try:
            result = self.fn()
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
        self._cleanup(task_id)

    def shutdown(self) -> None:
        for task_id in list(self._tasks.keys()):
            self._cleanup(task_id)

    def _handle_finished(self, task_id: str, result: Any) -> None:
        callback = self._on_done.pop(task_id, None)
        self._on_error.pop(task_id, None)
        self._cleanup(task_id)
        if callback:
            callback(result)

    def _handle_error(self, task_id: str, error_msg: str) -> None:
        callback = self._on_error.pop(task_id, None)
        self._on_done.pop(task_id, None)
        self._cleanup(task_id)
        if callback:
            callback(error_msg)

    def _handle_timeout(self, task_id: str) -> None:
        callback = self._on_error.pop(task_id, None)
        self._on_done.pop(task_id, None)
        self._cleanup(task_id)
        if callback:
            callback("任务超时")

    def _cleanup(self, task_id: str) -> None:
        worker = self._tasks.pop(task_id, None)
        if worker and worker.isRunning():
            worker.quit()
            worker.wait(1000)
        timer = self._timers.pop(task_id, None)
        if timer:
            timer.stop()
