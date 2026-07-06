import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
except ImportError as exc:
    raise unittest.SkipTest("PyQt6 is required for TensorRT export UI tests") from exc

from yolo26_app.core.predictor import YOLOPredictor
from yolo26_app.ui.export_dialog import ExportDialog


CALIBRATION_YAML = Path(__file__).parent / "fixtures" / "tensorrt_calibration.yaml"


class TensorRTExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.predictor = YOLOPredictor()
        self.predictor.model_path = "model.pt"

    def test_dialog_builds_current_int8_arguments(self):
        dialog = ExportDialog()
        dialog._format_combo.setCurrentIndex(dialog._format_combo.findData("engine"))
        dialog._precision_combo.setCurrentIndex(dialog._precision_combo.findData(8))
        dialog._data_edit.setText(str(CALIBRATION_YAML))
        dialog._fraction_spin.setValue(0.5)
        dialog._workspace_spin.setValue(0.0)
        emitted = []
        dialog.export_requested.connect(lambda fmt, kwargs: emitted.append((fmt, kwargs)))

        dialog._on_confirm()

        self.assertEqual(emitted[0][0], "engine")
        kwargs = emitted[0][1]
        self.assertEqual(kwargs["quantize"], 8)
        self.assertTrue(kwargs["dynamic"])
        self.assertTrue(dialog._dynamic_check.isChecked())
        self.assertFalse(dialog._dynamic_check.isEnabled())
        self.assertEqual(kwargs["device"], "0")
        self.assertEqual(kwargs["fraction"], 0.5)
        self.assertNotIn("workspace", kwargs)
        self.assertNotIn("half", kwargs)
        self.assertNotIn("int8", kwargs)
        dialog.deleteLater()

    def test_current_ultralytics_uses_quantize(self):
        with patch.object(self.predictor, "_ultralytics_supports_export_arg", return_value=True):
            kwargs = self.predictor._prepare_export_kwargs(
                "engine",
                {"quantize": 16, "device": "0"},
            )
        self.assertEqual(kwargs["quantize"], 16)
        self.assertNotIn("half", kwargs)

    def test_legacy_ultralytics_falls_back_to_old_flags(self):
        with patch.object(self.predictor, "_ultralytics_supports_export_arg", return_value=False):
            fp16 = self.predictor._prepare_export_kwargs(
                "engine",
                {"quantize": 16, "device": "0"},
            )
            int8 = self.predictor._prepare_export_kwargs(
                "engine",
                {
                    "quantize": 8,
                    "data": str(CALIBRATION_YAML),
                    "fraction": 0.5,
                    "device": "0",
                },
            )
        self.assertTrue(fp16["half"])
        self.assertTrue(int8["int8"])
        self.assertTrue(int8["dynamic"])
        self.assertNotIn("fraction", int8)

    def test_tensorrt_rejects_non_pytorch_source(self):
        self.predictor.model_path = "model.onnx"
        with self.assertRaisesRegex(RuntimeError, r"\.pt"):
            self.predictor._validate_tensorrt_environment({"device": "0"})

    def test_tensorrt_rejects_cpu_only_environment(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA GPU"):
                self.predictor._validate_tensorrt_environment({"device": "0"})

    def test_apply_tensorrt_enum_compat_adds_legacy_aliases(self):
        fake_trt = types.ModuleType("tensorrt")

        class FakeBuilderFlag:
            kFP16 = 1
            kINT8 = 2
            kTF32 = 4

        fake_trt.BuilderFlag = FakeBuilderFlag
        with patch.dict(sys.modules, {"tensorrt": fake_trt}):
            result = YOLOPredictor._apply_tensorrt_enum_compat()
        self.assertTrue(result)
        self.assertEqual(FakeBuilderFlag.FP16, FakeBuilderFlag.kFP16)
        self.assertEqual(FakeBuilderFlag.INT8, FakeBuilderFlag.kINT8)
        self.assertEqual(FakeBuilderFlag.TF32, FakeBuilderFlag.kTF32)

    def test_apply_tensorrt_enum_compat_skips_existing(self):
        fake_trt = types.ModuleType("tensorrt")

        class FakeBuilderFlag:
            FP16 = 999
            kFP16 = 1
            kINT8 = 2
            kTF32 = 4

        fake_trt.BuilderFlag = FakeBuilderFlag
        with patch.dict(sys.modules, {"tensorrt": fake_trt}):
            result = YOLOPredictor._apply_tensorrt_enum_compat()
        self.assertTrue(result)
        self.assertEqual(FakeBuilderFlag.FP16, 999)

    def test_apply_tensorrt_enum_compat_returns_false_without_tensorrt(self):
        with patch.dict(sys.modules, {"tensorrt": None}):
            result = YOLOPredictor._apply_tensorrt_enum_compat()
        self.assertFalse(result)

    def test_export_model_translates_builderflag_error(self):
        self.predictor.model = MagicMock()
        self.predictor.model.export.side_effect = AttributeError(
            "type object 'tensorrt_bindings.tensorrt.BuilderFlag' has no attribute 'FP16'"
        )
        with patch.object(self.predictor, "_validate_tensorrt_environment"), \
                patch.object(self.predictor, "_prepare_export_kwargs", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "不兼容"):
                self.predictor.export_model("engine", "/tmp/out", quantize=16, device="0")

    def test_export_model_reraises_unrelated_attribute_error(self):
        self.predictor.model = MagicMock()
        self.predictor.model.export.side_effect = AttributeError("totally unrelated error")
        with patch.object(self.predictor, "_validate_tensorrt_environment"), \
                patch.object(self.predictor, "_prepare_export_kwargs", return_value={}):
            with self.assertRaises(AttributeError):
                self.predictor.export_model("engine", "/tmp/out", quantize=16, device="0")


if __name__ == "__main__":
    unittest.main()
