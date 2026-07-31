import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yolo26_app.core.auto_annotator import YOLOPreAnnotator
from yolo26_app.core.model_session import ModelSession
from yolo26_app.core.predictor import YOLOPredictor
from yolo26_app.ui.annotation import _BatchDetectWorker, _YoloSamBatchWorker


class _DetectionBox:
    def __init__(self) -> None:
        self.xyxy = np.array([[1.0, 1.0, 6.0, 6.0]])
        self.cls = np.array([0.0])


class _DetectionResult:
    def __init__(self) -> None:
        self.boxes = [_DetectionBox()]
        self.masks = None


class _ArrayAdapter:
    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _BatchBoxes:
    def __init__(self) -> None:
        self.xyxy = _ArrayAdapter([[1.0, 1.0, 6.0, 6.0]])
        self.cls = _ArrayAdapter([0.0])

    def __len__(self) -> int:
        return 1


class _BatchResult:
    def __init__(self) -> None:
        self.boxes = _BatchBoxes()


class _FakeSamPredictor:
    def set_image(self, image) -> None:
        self.image_shape = image.shape

    def predict(self, **kwargs):
        mask = np.zeros((1, 8, 8), dtype=bool)
        mask[:, 1:7, 1:7] = True
        return mask, np.array([1.0]), None


def test_model_session_serializes_concurrent_access():
    session = ModelSession()
    barrier = threading.Barrier(2)
    active = 0
    peak_active = 0
    active_lock = threading.Lock()

    def access_model():
        nonlocal active, peak_active
        barrier.wait()
        with session.lock:
            with active_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.02)
            with active_lock:
                active -= 1

    threads = [threading.Thread(target=access_model) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak_active == 1


def test_model_session_tracks_loaded_model_metadata():
    session = ModelSession()
    model = object()

    session.set_model(model, task="detect")
    session.set_state("predicting", device="0")

    assert session.model is model
    assert session.task == "detect"
    assert session.state == "predicting"
    assert session.device == "0"

    session.set_state("idle", device="")

    assert session.device == ""


def test_preannotator_uses_model_from_shared_session(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

    class FakeModel:
        def predict(self, **kwargs):
            return [_DetectionResult()]

    session = ModelSession()
    session.set_model(FakeModel(), task="detect")
    annotator = YOLOPreAnnotator()
    annotator.set_model_session(session)

    annotations = annotator.annotate(str(image_path))

    assert annotator.has_model
    assert len(annotations) == 1
    assert annotations[0].item_type == "rect"


def test_normal_batch_worker_uses_shared_session_model(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

    class FakeModel:
        def predict(self, **kwargs):
            return [_DetectionResult()]

    session = ModelSession()
    session.set_model(FakeModel(), task="detect")
    annotator = YOLOPreAnnotator()
    annotator.set_model_session(session)
    worker = _BatchDetectWorker([str(image_path)], annotator, 0.25)
    completed = []
    worker.done_signal.connect(lambda results, total: completed.append((results, total)))

    worker.run()

    assert len(completed) == 1
    results, total = completed[0]
    assert total == 1
    assert str(image_path) in results


def test_predictor_and_preannotator_share_one_model_lock():
    active = 0
    peak_active = 0
    counter_lock = threading.Lock()
    barrier = threading.Barrier(2)

    class SlowModel:
        def predict(self, **kwargs):
            nonlocal active, peak_active
            with counter_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.03)
            with counter_lock:
                active -= 1
            return []

    model = SlowModel()
    predictor = YOLOPredictor()
    predictor.model = model
    predictor.session.set_model(model, task="detect")
    annotator = YOLOPreAnnotator()
    annotator.set_model_session(predictor.session)

    def annotate_predict():
        barrier.wait()
        annotator.predict_results(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_page_predict():
        barrier.wait()
        predictor.predict_frame(np.zeros((8, 8, 3), dtype=np.uint8))

    threads = [
        threading.Thread(target=annotate_predict),
        threading.Thread(target=test_page_predict),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert peak_active == 1


def test_yolo_sam_batch_worker_uses_shared_session_model(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(image_path), np.zeros((8, 8, 3), dtype=np.uint8))

    class FakeModel:
        def predict(self, **kwargs):
            return [_BatchResult()]

    session = ModelSession()
    session.set_model(FakeModel(), task="detect")
    annotator = YOLOPreAnnotator()
    annotator.set_model_session(session)
    worker = _YoloSamBatchWorker(
        [str(image_path)],
        annotator,
        _FakeSamPredictor(),
        0.25,
    )
    completed = []
    worker.done_signal.connect(lambda results, total: completed.append((results, total)))

    worker.run()

    assert len(completed) == 1
    results, total = completed[0]
    assert total == 1
    assert str(image_path) in results
    assert results[str(image_path)][0].item_type == "polygon"
