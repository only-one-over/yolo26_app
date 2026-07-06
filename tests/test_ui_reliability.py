import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QApplication, QWidget
except ImportError as exc:
    raise unittest.SkipTest("PyQt6 is required for UI reliability tests") from exc

from yolo26_app.core.config import ProjectConfig
from yolo26_app.ui.annotate_widget import AnnotateWidget
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
            "yolo26_app.ui.annotate_widget": _fake_module(
                "yolo26_app.ui.annotate_widget", "AnnotateWidget", _FakeAnnotateWidget
            ),
            "yolo26_app.ui.train_widget": _fake_module(
                "yolo26_app.ui.train_widget", "TrainWidget", _FakeTrainWidget
            ),
            "yolo26_app.ui.test_widget": _fake_module(
                "yolo26_app.ui.test_widget", "TestWidget", _FakeTestWidget
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


if __name__ == "__main__":
    unittest.main()
