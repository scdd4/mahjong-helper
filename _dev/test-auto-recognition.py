"""Deterministic screenshot integration checks, with no user calibration.

The synthetic fixtures paste upstream template art and verify plumbing,
localization, resizing and refusal behavior. They do not measure accuracy on
unseen real screenshots. The existing real ss002 image has an open North meld
and must be refused by the concealed-hand-only advice path.
"""
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / ".python-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".python-deps"))

from PIL import Image, ImageFilter
from auto_recognize import RecognitionError, _assess_hand, recognize_hand
from auto_vision_vendor import split as image_splitter

NAMES = ["1m", "2m", "3m", "4m", "0m", "6m", "1p", "2p", "3p", "7s", "8s", "9s", "1z", "1z"]


def fixture(count=14):
    image = Image.new("RGB", (1280, 720), (35, 55, 73))
    for index, name in enumerate(NAMES[:count]):
        with Image.open(ROOT / "auto_vision_vendor" / "templates" / (name + ".png")) as tile:
            tile = tile.convert("RGB").resize((60, 102), Image.Resampling.LANCZOS)
            gap = 20 if count == 14 and index == 13 else 0
            image.paste(tile, (100 + index * 63 + gap, 605))
    return image


class AutomaticRecognitionTests(unittest.TestCase):
    def assert_hand(self, image, count):
        result = recognize_hand(image)
        self.assertTrue(result["ok"])
        self.assertEqual([t["name"] for t in result["recognized_tiles"]], NAMES[:count])
        self.assertEqual(len(result["tiles"]), count)
        self.assertTrue(all(type(tile) is int and 0 <= tile <= 33 for tile in result["tiles"]))
        self.assertEqual(result["red"], [True, False, False])
        self.assertGreaterEqual(result["confidence"], 0.60)
        self.assertEqual(image_splitter._FRAME_CACHE, {})
        for x, y, width, height in result["boxes"]:
            self.assertTrue(0 <= x < x + width <= image.width)
            self.assertTrue(0 <= y < y + height <= image.height)
        return result

    def test_13_concealed_tiles(self):
        self.assert_hand(fixture(13), 13)

    def test_14_tiles_including_red_five_and_drawn_tile(self):
        result = self.assert_hand(fixture(), 14)
        self.assertTrue(result["recognized_tiles"][-1]["drawn"])

    def test_1080p_resizing(self):
        self.assert_hand(fixture().resize((1920, 1080), Image.Resampling.BICUBIC), 14)

    def test_changed_16_9_window_dimensions(self):
        self.assert_hand(fixture().resize((1440, 810), Image.Resampling.BICUBIC), 14)

    def test_black_outer_border_and_shifted_viewport(self):
        bordered = Image.new("RGB", (1600, 1000), "black")
        bordered.paste(fixture(), (160, 140))
        result = self.assert_hand(bordered, 14)
        self.assertEqual(result["canvas"], [160, 140, 1440, 860])
        self.assertGreaterEqual(result["boxes"][0][0], 260)

    def test_real_open_hand_is_refused(self):
        path = ROOT / "_dev" / "real" / "ss002.jpg"
        if not path.is_file():
            self.skipTest("Existing open-hand screenshot is not available")
        with Image.open(path) as source:
            with self.assertRaises(RecognitionError) as caught:
                recognize_hand(source)
        self.assertEqual(caught.exception.code, "open_hand")
        self.assertEqual(caught.exception.details["tile_count"], 11)
        self.assertEqual(caught.exception.details["meld_tiles"], 3)
        self.assertEqual(caught.exception.details["hand"], "")

    def test_non_table_images_are_refused(self):
        for color in ("black", "white", "#333333"):
            with self.subTest(color=color):
                with self.assertRaises(RecognitionError):
                    recognize_hand(Image.new("RGB", (1280, 720), color))

    def test_missing_tile_is_refused(self):
        with self.assertRaises(RecognitionError) as caught:
            recognize_hand(fixture(12))
        self.assertEqual(caught.exception.code, "invalid_count")

    def test_blurred_tiles_are_refused(self):
        with self.assertRaises(RecognitionError):
            recognize_hand(fixture().filter(ImageFilter.GaussianBlur(9)))

    def test_score_and_margin_safety_gates(self):
        good = recognize_hand(fixture())
        raw = {"tiles": [{"name": t["name"], "score": 0.55, "quad": [10 + i * 60, 500, 60 + i * 60, 600]}
                         for i, t in enumerate(good["recognized_tiles"])], "meld_tiles": 0}
        low = _assess_hand(raw, time.perf_counter())
        self.assertFalse(low["ok"])
        self.assertEqual(low["code"], "low_confidence")
        self.assertEqual(low["hand"], "")
        for tile in raw["tiles"]:
            tile.update(score=0.85, margin=0.005)
        ambiguous = _assess_hand(raw, time.perf_counter())
        self.assertFalse(ambiguous["ok"])
        self.assertEqual(ambiguous["code"], "low_confidence")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--export-fixture":
        destination = Path(sys.argv[2])
        destination.parent.mkdir(parents=True, exist_ok=True)
        fixture().save(destination)
        print(destination)
    else:
        unittest.main(verbosity=2)
