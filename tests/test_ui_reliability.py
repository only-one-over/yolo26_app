import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget
except ImportError as exc:
    raise unittest.SkipTest("PyQt6 is required for UI reliability tests") from exc

from yolo26_app.core.config import ProjectConfig
from yolo26_app.ui.annotation import AnnotateWidget, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from yolo26_app.ui.main_window import MainWindow


class _FakeAnnotateWidget(QWidget):
    state_changed = pyqtSignal()

    def set_project_config(self, config):
        self.config = config

    def set_yolo_model(self, model):
        self.model = model

    def flush_autosave(self):
        pass


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

    def test_import_directory_adds_images_and_delegates_videos_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            image = root / "a.jpg"
            nested_image = nested / "b.webp"
            video = nested / "clip.mp4"
            image.touch()
            nested_image.touch()
            video.touch()

            widget = AnnotateWidget()
            with patch.object(QFileDialog, "getExistingDirectory", return_value=str(root)), \
                    patch.object(widget, "_copy_to_project_images", side_effect=lambda path: Path(path).name), \
                    patch.object(widget, "_import_video_files", return_value=(1, 2, [])) as video_import, \
                    patch.object(widget, "_finish_media_import") as finish_import, \
                    patch.object(QMessageBox, "information") as info_message:
                widget._import_directory()

            self.assertEqual(widget._image_list, ["a.jpg", "b.webp"])
            self.assertEqual([Path(path).name for path in video_import.call_args.args[0]], ["clip.mp4"])
            finish_import.assert_called_once_with(select_first_if_empty=True)
            info_message.assert_called_once()
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
