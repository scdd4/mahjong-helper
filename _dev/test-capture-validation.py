"""离线验证截图输入、学习会话和坏模板；禁止真实屏幕捕获。"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("capture_engine", ROOT / "capture.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


class CaptureValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mj-validation-")
        self.root = Path(self.tmp.name)
        engine.TPL_DIR = str(self.root / "templates")
        engine.CELL_DIR = str(self.root / "cells")
        engine.CALIB = str(self.root / "calib.json")
        engine.SHOT = str(self.root / "latest.png")
        self.no_capture = mock.patch.object(engine, "grab", side_effect=AssertionError("不允许真实截屏"))
        self.no_capture.start()
        self.image = ROOT / "_dev" / "mock" / "screenA.png"

    def tearDown(self):
        self.no_capture.stop()
        self.tmp.cleanup()

    def scan(self, **kwargs):
        return engine.dispatch({"mode": "scan", "image": str(self.image), "box": [280, 780, 1120, 110], "n": 14, **kwargs})

    def test_missing_image_never_falls_back_to_screen(self):
        with self.assertRaisesRegex(ValueError, "未调用屏幕截图"):
            engine.dispatch({"mode": "probe", "image": str(self.root / "missing.png")})

    def test_default_capture_does_not_save_full_desktop(self):
        with mock.patch.object(engine, "grab", return_value=Image.new("RGB", (32, 32))), mock.patch.dict(os.environ, {"MJ_SAVE_SHOT": "0"}):
            engine.dispatch({"mode": "probe"})
        self.assertFalse(Path(engine.SHOT).exists())

    def test_invalid_count_rejected_before_screen_access(self):
        for value in (0, -1, 15, 1.5, True, "14"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                engine.dispatch({"mode": "scan", "n": value, "box": [0, 0, 140, 50]})

    def test_invalid_geometry_rejected(self):
        for box in ([0, 0, 0, 1], [0, 0, float("inf"), 50], [-1, 0, 100, 50], [0, 0, 100000, 50], "bad", [0, 0], [], False):
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.scan(box=box)

    def test_invalid_discard_side_rejected(self):
        with self.assertRaises(ValueError):
            self.scan(discardBoxes={"evil": [0, 0, 100, 100]})

    def test_learn_requires_matching_session(self):
        first = self.scan()
        second = self.scan()
        self.assertNotEqual(first["scanId"], second["scanId"])
        for scan_id in (None, first["scanId"], "arbitrary"):
            with self.subTest(scan_id=scan_id), self.assertRaisesRegex(ValueError, "过期"):
                engine.dispatch({"mode": "learn", "assign": {"0": 0}, "scanId": scan_id})
        result = engine.dispatch({"mode": "learn", "assign": {"0": 0}, "scanId": second["scanId"]})
        self.assertEqual(result["saved"], 1)
        self.assertEqual(set(engine.load_templates()), {0})

    def test_invalid_assignment_does_not_partially_write(self):
        scan_id = self.scan()["scanId"]
        for assign in ({"0": 0, "1": -1}, {"0": 0, "1": 34}, {"0": 0, "14": 1},
                       {"0": 0, "1": True}, {"0": 0, "1": "1"}, [], {"../0": 0}):
            with self.subTest(assign=assign), self.assertRaises(ValueError):
                engine.dispatch({"mode": "learn", "scanId": scan_id, "assign": assign})
        self.assertEqual(engine.load_templates(), {})

    def test_wrong_count_cannot_learn(self):
        scan_id = self.scan()["scanId"]
        with self.assertRaises(ValueError):
            engine.dispatch({"mode": "learn", "scanId": scan_id, "assign": {"0": 0}, "n": 13})

    def test_broken_and_out_of_range_templates_are_ignored(self):
        os.makedirs(engine.TPL_DIR)
        np.save(Path(engine.TPL_DIR) / "0.npy", np.ones(3))
        np.save(Path(engine.TPL_DIR) / "1.npy", np.full(1120, float("nan")))
        np.save(Path(engine.TPL_DIR) / "34.npy", np.ones(1120))
        (Path(engine.TPL_DIR) / "2.npy").write_bytes(b"broken")
        self.assertEqual(engine.load_templates(), {})

    def test_fewer_than_three_discard_hits_are_not_reported(self):
        scores = np.zeros((24, 2), dtype=np.float32)
        scores[:2, 0] = 0.95
        with mock.patch.object(engine, "_batch_scores", return_value=([0, 1], scores)):
            result = engine.scan_discards(Image.new("RGB", (600, 400)), {"top": [0, 0, 600, 400]}, {0: np.ones(1120)})
        self.assertEqual(result, {})

    def test_ambiguous_discard_hits_are_not_reported(self):
        scores = np.full((24, 2), 0.95, dtype=np.float32)
        with mock.patch.object(engine, "_batch_scores", return_value=([0, 1], scores)):
            result = engine.scan_discards(Image.new("RGB", (600, 400)), {"top": [0, 0, 600, 400]}, {0: np.ones(1120)})
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
