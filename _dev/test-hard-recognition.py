"""高难牌型的视觉识别专项：合成图，不依赖真实截图。

检测极端手牌（四同牌/多赤五/字牌密集/幺九13种）在模板匹配下是否
稳定识别，以及缩放/噪声边界是否触发安全拒绝而非错误输出。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.python-deps'))

from PIL import Image, ImageFilter
from auto_recognize import recognize_hand, RecognitionError

TEMPLATES = ROOT / 'auto_vision_vendor' / 'templates'


def fixture(names, size=(1280, 720)):
    """把模板牌贴到标准手牌行；14 张时最后一张留出摸牌间隔。"""
    image = Image.new('RGB', size, (35, 55, 73))
    count = len(names)
    for index, name in enumerate(names):
        with Image.open(TEMPLATES / (name + '.png')) as tile:
            tile = tile.convert('RGB').resize((60, 102), Image.Resampling.LANCZOS)
            gap = 20 if count == 14 and index == 13 else 0
            x = 100 + index * 63 + gap
            if x + 60 > image.width:
                continue
            image.paste(tile, (x, size[1] - 115))
    return image


class HardRecognitionTests(unittest.TestCase):
    def assert_names(self, names, expected):
        result = recognize_hand(fixture(names))
        self.assertTrue(result['ok'], result)
        got = [t['name'] for t in result['recognized_tiles']]
        self.assertEqual(got, expected)

    def assert_safe_refusal(self, names, label):
        """灰区牌型必须被拒绝而非带错牌进入策略，且必须指出具体位置。"""
        with self.assertRaises(RecognitionError) as caught:
            recognize_hand(fixture(names))
        self.assertEqual(caught.exception.code, 'low_confidence', label)
        details = caught.exception.details
        self.assertTrue(details.get('uncertain_tiles'), label)
        self.assertIn('第', str(caught.exception), label)

    def test_four_identical_tiles(self):
        names = ['1m'] * 4 + ['2m','3m','4p','5p','6p','7s','8s','9s','1z','2z']
        self.assert_safe_refusal(names, '四同牌')

    def test_multiple_red_fives(self):
        names = ['0m','5m','0p','5p','0s','5s','1m','2m','3m','2p','3p','4p','1z','1z']
        self.assert_safe_refusal(names, '多赤五')

    def test_honor_heavy_hand(self):
        names = ['1z','2z','3z','4z','5z','6z','7z','7z','1m','2m','3m','9p','9p','9s']
        self.assert_safe_refusal(names, '字牌密集')

    def test_kokushi_13_unique_terminals_plus_pair(self):
        names = ['1m','9m','1p','9p','1s','9s','1z','2z','3z','4z','5z','6z','7z','7z']
        self.assert_safe_refusal(names, '国士幺九密集')

    def test_4k_scaling_never_misreads_honor_heavy_hand(self):
        # 4K 放大后字牌密集手牌可能落入低置信灰区；关键是绝不误报错误牌型。
        names = ['1z','2z','3z','4z','5z','6z','7z','7z','1m','2m','3m','9p','9p','9s']
        image = fixture(names).resize((3840, 2160), Image.Resampling.LANCZOS)
        try:
            result = recognize_hand(image)
        except RecognitionError as caught:
            self.assertEqual(caught.code, 'low_confidence')
            return
        self.assertEqual([t['name'] for t in result['recognized_tiles']], names)

    def test_mild_blur_is_refused(self):
        names = ['1m','2m','3m','4m','5m','6m','7p','8p','9p','2s','3s','4s','1z','1z']
        with self.assertRaises(RecognitionError) as caught:
            recognize_hand(fixture(names).filter(ImageFilter.GaussianBlur(1.2)))
        self.assertEqual(caught.exception.code, 'low_confidence')

    def test_heavy_blur_is_refused_not_misread(self):
        names = ['1m','2m','3m','4m','5m','6m','7p','8p','9p','2s','3s','4s','1z','1z']
        with self.assertRaises(RecognitionError):
            recognize_hand(fixture(names).filter(ImageFilter.GaussianBlur(9)))


if __name__ == '__main__':
    unittest.main()
