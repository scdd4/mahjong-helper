"""Offline regressions for full-desktop ROI recovery and settings persistence."""
from dataclasses import replace
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from desktop_runtime import Settings,save_settings,load_settings
from desktop_widgets import settings_from_form,MODE_LABELS
from table_viewport import locate_table,prepare_table_image
from scene_recognize import read_scene
from PIL import Image

EXPECTED=['1m','2m','4m','6m','6m','3p','6s','3z','4z','5z','6z','7z','7z']


class SettingsTests(unittest.TestCase):
    def form(self,**changes):
        values={'base_url':'http://127.0.0.1:8080/v1','model':'test-model',
                'api_key':'test-only-secret','reasoning_effort':'high','max_tokens':'32768',
                'timeout':'900','capture_source':'全屏','analysis_mode':MODE_LABELS['full']}
        values.update(changes)
        return values

    def test_form_saves_every_model_field_and_preserves_mini_preferences(self):
        initial=Settings(mini_mode=True,mini_topmost=False)
        settings=settings_from_form(initial,self.form())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            save_settings(settings,path)
            restored=load_settings(path)
            self.assertEqual(restored,settings)
            self.assertTrue(restored.mini_mode)
            self.assertFalse(restored.mini_topmost)
            self.assertEqual(restored.capture_source,'screen')
            self.assertEqual(restored.analysis_mode,'full')
            self.assertNotIn(settings.api_key,path.read_text(encoding='utf-8'))
        self.assertEqual(initial.model,Settings().model)

    def test_invalid_form_does_not_overwrite_an_existing_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json'
            initial=Settings(api_key='test-only-secret',mini_mode=True)
            save_settings(initial,path)
            before=path.read_bytes()
            with self.assertRaises(ValueError):
                settings_from_form(initial,self.form(base_url='not a URL'))
            self.assertEqual(path.read_bytes(),before)

    def test_legacy_configuration_defaults_and_mini_flags_validate(self):
        self.assertFalse(Settings().mini_mode)
        self.assertTrue(Settings().mini_topmost)
        with self.assertRaises(ValueError):replace(Settings(),mini_mode='yes').validate()


class FullDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=os.environ.get('MJ_DESKTOP_FIXTURE')
        if not folder:raise unittest.SkipTest('Set MJ_DESKTOP_FIXTURE to the private screenshot path')
        with Image.open(folder) as source:cls.desktop=source.convert('RGB')

    @classmethod
    def tearDownClass(cls):cls.desktop.close()

    def assert_hand(self,image):
        result=read_scene(image)
        self.assertEqual([t['name'] for t in result['hand']['tiles']],EXPECTED)
        self.assertTrue(result['hand']['reliable'])
        return result

    def test_desktop_without_window_metadata_is_located_and_read(self):
        self.assertNotIn('game_bbox',self.desktop.info)
        result=self.assert_hand(self.desktop)
        left,top,right,bottom=result['viewport']
        self.assertLess(abs(left-966),8)
        self.assertLess(abs(top-47),8)
        self.assertLess(abs(right-2554),8)
        self.assertLess(abs(bottom-891),8)
        self.assertGreater(result['hand']['tiles'][0]['box'][0],1100)

    def test_preview_and_analysis_share_the_same_cropped_table(self):
        original_info=dict(self.desktop.info)
        with prepare_table_image(self.desktop) as prepared:
            self.assertLess(prepared.width,1700)
            self.assertLess(prepared.height,900)
            self.assertEqual(prepared.info['game_bbox'],[0,0,prepared.width,prepared.height])
            self.assertEqual(prepared.info['viewport_source'],'visual')
            self.assert_hand(prepared)
        self.assertEqual(self.desktop.info,original_info)

    def test_bad_native_coordinates_fall_back_to_visual_location(self):
        with self.desktop.copy() as image:
            image.info['game_bbox']=[1800,10,3800,1200]
            box,source=locate_table(image)
            self.assertEqual(source,'visual')
            self.assertLess(abs(box[0]-966),8)

    def test_a_moved_and_resized_game_does_not_require_fixed_desktop_coordinates(self):
        with self.desktop.crop((966,47,2554,891)) as game:
            with game.resize((1280,680),Image.Resampling.LANCZOS) as smaller:
                with Image.new('RGB',(2300,1300),'#bddfe8') as desktop:
                    desktop.paste(smaller,(80,160))
                    self.assert_hand(desktop)

    def test_plain_desktop_does_not_invent_a_game_rectangle(self):
        with Image.new('RGB',(1920,1080),'#bddfe8') as image:
            self.assertEqual(locate_table(image),(None,'unlocated'))

    def test_existing_game_screenshots_are_not_trimmed_into_unreliable_hands(self):
        folder=os.environ.get('MJ_USER_FIXTURES')
        if not folder:self.skipTest('Set MJ_USER_FIXTURES to the existing private screenshots')
        expected=[['3m','6m','8m','6p','7p','8p','1s','0s','9s','1z','2z','5z','7z'],
                  ['1m','1m','2m','2m','4m','7m','8m','1p','3p','4p','4p','6p','8p','7m']]
        for i,hand in enumerate(expected,1):
            with Image.open(Path(folder)/('用户样例-%d.png'%i)) as source, prepare_table_image(source) as prepared:
                result=read_scene(prepared)
                self.assertEqual([t['name'] for t in result['hand']['tiles']],hand)
                self.assertTrue(result['hand']['reliable'])


if __name__=='__main__':unittest.main(verbosity=2)
