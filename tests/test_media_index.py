import json
import tempfile
import unittest
from pathlib import Path

from yolo26_app.core.media_index import INDEX_FILENAME, MediaIndex
from yolo26_app.core.startup_metrics import StartupMetrics


class MediaIndexTests(unittest.TestCase):
    def test_snapshot_upsert_and_clear_are_rebuildable(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            first = project / "first.jpg"
            second = project / "second.jpg"
            first.write_bytes(b"one")
            second.write_bytes(b"two")

            index = MediaIndex(project)
            index.sync_snapshot([str(first), str(second)], {str(first): 2})
            self.assertEqual(index.count(), 2)

            first.write_bytes(b"changed")
            index.upsert([str(first)], {str(first): 3})
            self.assertEqual(index.count(), 2)

            index.clear()
            self.assertEqual(index.count(), 0)

    def test_corrupt_index_is_recreated_without_touching_media(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            image = project / "image.jpg"
            image.write_bytes(b"image")
            (project / INDEX_FILENAME).write_text("not a database", encoding="utf-8")

            index = MediaIndex(project)
            index.sync_snapshot([str(image)], {})

            self.assertEqual(index.count(), 1)
            self.assertEqual(image.read_bytes(), b"image")

    def test_startup_metrics_writes_non_sensitive_event(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = StartupMetrics(Path(directory))
            metrics.mark("main_window_shown", media_count=12)
            records = (Path(directory) / "logs" / "startup_metrics.jsonl").read_text("utf-8").splitlines()
            event = json.loads(records[-1])
            self.assertEqual(event["stage"], "main_window_shown")
            self.assertEqual(event["media_count"], 12)
            self.assertNotIn("path", event)
