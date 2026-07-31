import json
import cv2
import numpy as np
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtGui import QCloseEvent
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget
except ImportError as exc:
    raise unittest.SkipTest("PyQt6 is required for UI reliability tests") from exc

from yolo26_app.core.config import ProjectConfig, TrainConfig
from yolo26_app.core.trainer import YOLOTrainer
from yolo26_app.ui.annotation import (
    AnnotateWidget,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    _ImageLoadWorker,
    _MediaImportWorker,
)
from yolo26_app.ui import inference as inference_ui
from yolo26_app.ui import training as training_ui
from yolo26_app.ui.main_window import MainWindow


class _FakeAnnotateWidget(QWidget):
    state_changed = pyqtSignal()

    def set_project_config(self, config):
        self.config = config

    def set_model_session(self, session):
        self.session = session

    def set_yolo_model(self, model):
        self.model = model

    def flush_autosave(self):
        pass

    def has_running_background_workers(self):
        return False

    def request_stop_background_workers(self):
        pass

    def stop_background_threads(self):
        return True


class _FakeTrainWidget(QWidget):
    def set_project_config(self, config):
        self.config = config


class _FakeTestWidget(QWidget):
    model_loaded = pyqtSignal(object)

    def set_project_config(self, config):
        self.config = config


def _fake_module(name, class_name, widget_class):
    module = types.ModuleType(name)
    setattr(module, class_name, widget_class)
    return module


class _FakeInferenceWorker(QWidget):
    result_signal = pyqtSignal(object, object)
    finished = pyqtSignal()

    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.start_calls = 0
        self.stop_calls = 0
        self.submit_calls = 0
        self._running = False
        self._busy = False

    @property
    def is_busy(self):
        return self._busy

    def isRunning(self):
        return self._running

    def start(self):
        self.start_calls += 1
        self._running = True

    def stop(self):
        self.stop_calls += 1

    def submit(self, *args, **kwargs):
        self.submit_calls += 1


class _FakeCapture:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True

class UiReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_lazy_pages_do_not_depend_on_creation_order(self):
        fake_modules = {
            "yolo26_app.ui.annotation": _fake_module(
                "yolo26_app.ui.annotation", "AnnotateWidget", _FakeAnnotateWidget
            ),
            "yolo26_app.ui.training": _fake_module(
                "yolo26_app.ui.training", "TrainWidget", _FakeTrainWidget
            ),
            "yolo26_app.ui.inference": _fake_module(
                "yolo26_app.ui.inference", "TestWidget", _FakeTestWidget
            ),
        }
        with patch.dict(sys.modules, fake_modules):
            window = MainWindow()
            window._switch_page(2)
            self.assertIs(window.stacked.currentWidget(), window.test_widget)
            window._switch_page(1)
            self.assertIs(window.stacked.currentWidget(), window.train_widget)
            window._switch_page(2)
            self.assertIs(window.stacked.currentWidget(), window.test_widget)
            window.deleteLater()

    def test_training_reports_actual_save_dir_at_train_start(self):
        trainer = YOLOTrainer(TrainConfig(), "")
        reported_dirs = []
        trainer.save_dir_signal.connect(reported_dirs.append)
        fake_ultralytics_trainer = types.SimpleNamespace(save_dir=Path("runs/train2"))

        trainer._on_train_start(fake_ultralytics_trainer)

        self.assertEqual(reported_dirs, [str(Path("runs/train2"))])
    def test_realtime_inference_restarts_with_a_new_worker_after_stop(self):
        first_capture = _FakeCapture()
        second_capture = _FakeCapture()
        with patch("yolo26_app.ui.inference._InferenceWorker", _FakeInferenceWorker), \
                patch("yolo26_app.ui.inference.cv2.VideoCapture", side_effect=[first_capture, second_capture]):
            widget = inference_ui.TestWidget()
            widget.predictor.model = object()

            widget._start_capture("first.mp4")
            first_worker = widget._inference_worker
            self.assertIsNotNone(first_worker)
            self.assertEqual(first_worker.start_calls, 1)

            widget._on_stop()
            self.assertIsNone(widget._inference_worker)
            self.assertEqual(first_worker.stop_calls, 1)

            widget._start_capture("second.mp4")
            second_worker = widget._inference_worker
            self.assertIsNotNone(second_worker)
            self.assertIsNot(first_worker, second_worker)
            self.assertEqual(second_worker.start_calls, 1)

            widget._on_timer_timeout()
            self.assertEqual(second_worker.submit_calls, 1)
            widget._on_stop()
            widget.deleteLater()

    def test_model_loaded_emits_shared_session(self):
        widget = inference_ui.TestWidget()
        received_sessions = []
        widget.model_loaded.connect(received_sessions.append)

        with patch.object(QMessageBox, "information"):
            widget._on_model_loaded(
                True,
                "model.pt",
                {"task": "detect", "class_names": []},
            )

        self.assertEqual(received_sessions, [widget.predictor.session])
        widget.deleteLater()

    def test_close_event_blocks_while_worker_is_running(self):
        widget = inference_ui.TestWidget()
        worker = MagicMock()
        worker.isRunning.return_value = True
        widget._validate_worker = worker
        event = QCloseEvent()

        with patch.object(QMessageBox, "warning") as warning:
            widget.closeEvent(event)

        worker.request_stop.assert_called_once_with()
        warning.assert_called_once()
        self.assertFalse(event.isAccepted())
        widget._validate_worker = None
        widget.deleteLater()

    def test_training_close_event_blocks_while_worker_is_running(self):
        widget = training_ui.TrainWidget()
        worker = MagicMock()
        worker.isRunning.return_value = True
        widget._trainer = worker
        event = QCloseEvent()

        with patch.object(QMessageBox, "warning") as warning:
            widget.closeEvent(event)

        worker.stop.assert_called_once_with()
        warning.assert_called_once()
        self.assertFalse(event.isAccepted())
        widget._trainer = None
        widget.deleteLater()

    def test_annotation_stop_preserves_reference_until_worker_finishes(self):
        widget = AnnotateWidget()
        worker = MagicMock()
        worker.isRunning.return_value = True
        widget._batch_worker = worker

        stopped = widget.stop_background_threads()

        worker.stop.assert_called_once_with()
        self.assertFalse(stopped)
        self.assertTrue(widget.has_running_background_workers())
        self.assertIs(widget._batch_worker, worker)
        widget._batch_worker = None
        widget.deleteLater()

    def test_main_window_tracks_and_stops_annotation_workers(self):
        annotate_widget = MagicMock()
        annotate_widget.has_running_background_workers.return_value = True
        host = types.SimpleNamespace(
            train_widget=None,
            test_widget=None,
            annotate_widget=annotate_widget,
        )

        self.assertTrue(MainWindow._has_running_tasks(host))
        MainWindow._stop_all_tasks(host)

        annotate_widget.request_stop_background_workers.assert_called_once_with()

    def test_main_window_tracks_startup_diagnostic_workers(self):
        diagnostic_worker = MagicMock()
        diagnostic_worker.isRunning.return_value = True
        host = types.SimpleNamespace(
            train_widget=None,
            test_widget=None,
            annotate_widget=None,
            _env_check_worker=diagnostic_worker,
            _gpu_detect_worker=None,
        )

        self.assertTrue(MainWindow._has_running_tasks(host))
        MainWindow._stop_all_tasks(host)

        diagnostic_worker.requestInterruption.assert_called_once_with()

    def test_space_navigation_advances_one_image(self):
        widget = AnnotateWidget()
        widget._add_image_item("first.jpg")
        widget._add_image_item("second.jpg")
        widget._image_list = ["first.jpg", "second.jpg"]
        widget._annotations_dict = {"first.jpg": [], "second.jpg": []}
        widget._image_list_widget.setCurrentRow(0)

        widget._go_to_next_image()
        self.assertEqual(widget._image_list_widget.currentRow(), 1)
        widget._go_to_next_image()
        self.assertEqual(widget._image_list_widget.currentRow(), 1)
        widget.deleteLater()

    def test_project_autosave_is_complete_json(self):
        annotations_path = Path(__file__).parent / "annotations.json"
        try:
            project = ProjectConfig(
                project_name="test",
                project_path=str(annotations_path.parent),
            )
            host = MainWindow()
            host.current_project_config = project
            widget = host._ensure_widget(0)
            widget._image_list = ["image.jpg"]
            widget._annotations_dict = {"image.jpg": []}
            widget._current_image_path = "image.jpg"

            self.assertTrue(widget._save_annotations_to_project())
            saved = json.loads(annotations_path.read_text("utf-8"))
            self.assertEqual(saved["image_list"], ["image.jpg"])
            self.assertEqual(saved["current_image_path"], "image.jpg")
            host.deleteLater()
        finally:
            annotations_path.unlink(missing_ok=True)

    def test_temporary_state_restores_classes_and_current_image(self):
        source = AnnotateWidget()
        source.get_label_manager().add_class("target")
        source._image_list = ["first.jpg", "second.jpg"]
        source._annotations_dict = {"first.jpg": [], "second.jpg": []}
        source._current_image_path = "second.jpg"
        state = source.save_state()

        restored = AnnotateWidget()
        restored.restore_state(state)
        self.assertEqual(
            [item.name for item in restored.get_label_manager().get_all_classes()],
            ["target"],
        )
        self.assertEqual(restored._image_list_widget.currentRow(), 1)
        self.assertEqual(restored._current_image_path, "second.jpg")
        if restored._thumb_worker is not None:
            restored._thumb_worker.wait(3000)
        source.deleteLater()
        restored.deleteLater()

    def test_unprojected_annotations_are_written_to_recovery_file(self):
        recovery_path = Path(__file__).parent / "app_state.json"
        try:
            with patch("yolo26_app.ui.main_window.APP_STATE_FILE", recovery_path):
                host = MainWindow()
                widget = host._ensure_widget(0)
                widget.get_label_manager().add_class("temporary")
                widget._image_list = ["image.jpg"]
                widget._annotations_dict = {"image.jpg": []}
                widget._current_image_path = "image.jpg"

                host._save_app_state()
                saved = json.loads(recovery_path.read_text("utf-8"))
                self.assertEqual(
                    saved["annotate_state"]["classes"][0]["name"],
                    "temporary",
                )
                self.assertEqual(
                    saved["annotate_state"]["current_image_path"],
                    "image.jpg",
                )
                host.deleteLater()
        finally:
            recovery_path.unlink(missing_ok=True)


    def test_media_directory_scan_recurses_supported_images_and_videos(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            for idx, ext in enumerate(sorted(IMAGE_EXTENSIONS)):
                target_dir = nested if idx % 2 else root
                (target_dir / f"image_{idx}{ext}").touch()
            for idx, ext in enumerate(sorted(VIDEO_EXTENSIONS)):
                target_dir = nested if idx % 2 else root
                (target_dir / f"video_{idx}{ext}").touch()
            (nested / "ignore.txt").touch()

            widget = AnnotateWidget()
            image_files, video_files = widget._scan_media_directory(str(root))

            self.assertEqual(len(image_files), len(IMAGE_EXTENSIONS))
            self.assertEqual(len(video_files), len(VIDEO_EXTENSIONS))
            self.assertTrue(all(Path(path).suffix.lower() in IMAGE_EXTENSIONS for path in image_files))
            self.assertTrue(all(Path(path).suffix.lower() in VIDEO_EXTENSIONS for path in video_files))
            widget.deleteLater()

    def test_media_import_worker_scans_and_copies_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "source"
            output = Path(temp_dir) / "output"
            nested = root / "nested"
            nested.mkdir(parents=True)
            (root / "a.jpg").write_bytes(b"a")
            (nested / "b.webp").write_bytes(b"b")
            completed = []
            worker = _MediaImportWorker(
                directory_path=str(root),
                output_dir=str(output),
                video_output_dir=str(output),
            )
            worker.done_signal.connect(
                lambda images, frames, videos, failed, cancelled: completed.append(
                    (images, frames, videos, failed, cancelled)
                )
            )

            worker.run()

            self.assertEqual(len(completed), 1)
            images, frames, videos, failed, cancelled = completed[0]
            self.assertEqual({Path(path).name for path in images}, {"a.jpg", "b.webp"})
            self.assertEqual(frames, [])
            self.assertEqual(videos, 0)
            self.assertEqual(failed, [])
            self.assertFalse(cancelled)

    def test_media_import_worker_extracts_at_configured_interval(self):
        class FakeCapture:
            def __init__(self):
                self.index = 0
                self.released = False

            def isOpened(self):
                return True

            def get(self, _property):
                return 2.0

            def read(self):
                if self.index >= 5:
                    return False, None
                self.index += 1
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "clip.mp4"
            video.touch()
            output = Path(temp_dir) / "frames"
            completed = []
            capture = FakeCapture()
            worker = _MediaImportWorker(
                video_paths=[str(video)],
                video_output_dir=str(output),
                frame_interval_seconds=1.0,
            )
            worker.done_signal.connect(
                lambda images, frames, videos, failed, cancelled: completed.append(
                    (images, frames, videos, failed, cancelled)
                )
            )

            with patch("yolo26_app.ui.annotation.cv2.VideoCapture", return_value=capture):
                worker.run()

            self.assertEqual(len(completed), 1)
            images, frames, videos, failed, cancelled = completed[0]
            self.assertEqual(images, [])
            self.assertEqual(len(frames), 3)
            self.assertEqual(videos, 1)
            self.assertEqual(failed, [])
            self.assertFalse(cancelled)
            self.assertTrue(capture.released)

    def test_image_load_worker_emits_qimage_not_qpixmap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "large.jpg"
            cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
            loaded = []
            worker = _ImageLoadWorker(str(image_path))
            worker.image_ready.connect(lambda path, image: loaded.append((path, image)))

            worker.run()

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0][0], str(image_path))
            self.assertFalse(loaded[0][1].isNull())

    def test_import_directory_starts_one_background_media_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            widget = AnnotateWidget()
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)), \
                    patch.object(widget, "_ask_video_frame_interval", return_value=2.0), \
                    patch.object(widget, "_start_media_import") as start_import:
                widget._import_directory()

            start_import.assert_called_once_with(
                directory_path=str(root),
                frame_interval_seconds=2.0,
                select_first_if_empty=True,
            )
            widget.deleteLater()

    def test_import_image_files_does_not_duplicate_existing_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "same.jpg"
            image.touch()

            widget = AnnotateWidget()
            widget._image_list = ["same.jpg"]
            widget._annotations_dict = {"same.jpg": []}
            with patch.object(widget, "_copy_to_project_images", return_value="same.jpg"):
                added = widget._import_image_files([str(image)])

            self.assertEqual(added, 0)
            self.assertEqual(widget._image_list, ["same.jpg"])
            widget.deleteLater()

    def test_video_frame_paths_are_unique_per_existing_file_and_video_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            widget = AnnotateWidget()

            first = widget._next_video_frame_path(output_dir, "clip.mp4", 0)
            first.touch()
            second = widget._next_video_frame_path(output_dir, "clip.mp4", 0)
            other = widget._next_video_frame_path(output_dir, "other.mp4", 0)

            self.assertEqual(first.name, "clip_frame_000000.jpg")
            self.assertEqual(second.name, "clip_frame_000000_1.jpg")
            self.assertEqual(other.name, "other_frame_000000.jpg")
            widget.deleteLater()

    def test_clear_imported_images_resets_workspace_after_confirmation(self):
        widget = AnnotateWidget()
        widget._image_list = ["first.jpg", "second.jpg"]
        widget._annotations_dict = {"first.jpg": [object()], "second.jpg": []}
        widget._current_image_path = "first.jpg"
        widget._add_image_item("first.jpg")
        widget._add_image_item("second.jpg")

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes), \
                patch.object(widget, "_schedule_autosave") as autosave:
            widget._clear_imported_images()

        self.assertEqual(widget._image_list, [])
        self.assertEqual(widget._annotations_dict, {})
        self.assertEqual(widget._current_image_path, "")
        self.assertEqual(widget._image_list_widget.count(), 0)
        autosave.assert_called_once()
        widget.deleteLater()

    def test_clear_imported_images_cancel_keeps_workspace(self):
        widget = AnnotateWidget()
        widget._image_list = ["first.jpg"]
        widget._annotations_dict = {"first.jpg": []}
        widget._current_image_path = "first.jpg"
        widget._add_image_item("first.jpg")

        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No), \
                patch.object(widget, "_schedule_autosave") as autosave:
            widget._clear_imported_images()

        self.assertEqual(widget._image_list, ["first.jpg"])
        self.assertEqual(widget._annotations_dict, {"first.jpg": []})
        self.assertEqual(widget._current_image_path, "first.jpg")
        self.assertEqual(widget._image_list_widget.count(), 1)
        autosave.assert_not_called()
        widget.deleteLater()

if __name__ == "__main__":
    unittest.main()
