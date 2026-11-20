"""Regression cases for strategy facts, readable names, streaming and I/O failures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch,MagicMock

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from desktop_runtime import (Settings,tile_name,tile_code,humanize_tiles,build_safety_facts,save_settings,load_settings)
from spark_strategy import SparkClient,hand_tile_total,validate_report
from table_strategy import analyze_table
from strategy_context import meld_description
from PIL import Image

HAND=['1m','2m','3m','4m','5m','6m','7m','8m','9m','1p','2p','3p','5z','6z']


class FactTests(unittest.TestCase):
    def test_chinese_tile_names_roundtrip_and_public_text_contains_no_codes(self):
        for code in ['1m','9p','6s','0s','7z']:
            self.assertEqual(tile_code(tile_name(code)),code)
        self.assertEqual(tile_name('6s'),'六条')
        self.assertEqual(humanize_tiles('打3m，留456s和0p'),'打三万，留四条、五条、六条和赤五筒')

    def test_kan_uses_three_effective_tiles_and_counts_before_draw(self):
        local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND[:10]]},
               'melds':{'own':{'groups':[[{'name':'7z'}]*4]}}}
        self.assertEqual(hand_tile_total(HAND[:10],local),13)
        report={'observed':{'hand':HAND[:10],'melds':{'own':[{'type':'kan','tiles':['7z']*4}]}},
                'strategy':{'recommendation':{'tile':'1m'}}}
        self.assertTrue(any('13' in warning for warning in validate_report(report,local)))

    def test_duplicate_discards_do_not_create_two_sided_suji(self):
        facts=build_safety_facts({'across':['4m','4m']},{'across':True})
        self.assertEqual(facts['两筋']['across'],[])
        both=build_safety_facts({'across':['1m','7m']},{'across':True})
        self.assertIn('4m',both['两筋']['across'])
        self.assertEqual(facts['客风客三元'],[])

    def test_genbutsu_is_per_opponent_and_red_five_is_normalized(self):
        facts=build_safety_facts({'right':['0p'],'across':['1m']},{'right':True,'across':True})
        self.assertEqual(facts['现物']['right'],['5p'])
        self.assertNotIn('5p',facts['现物']['across'])

    def test_chi_does_not_cross_suit_boundaries(self):
        from scene_recognize import _infer_chi_middle
        group=[{'name':'9m'},{'name':None},{'name':'2p'}]
        _infer_chi_middle(group)
        self.assertIsNone(group[1]['name'])

    def test_multi_rotation_agreement_is_accepted_without_unpack_error(self):
        import scene_recognize as scene
        import numpy as np
        image=np.zeros((50,30,3),dtype='uint8')
        with patch.object(scene,'_warp_tile',return_value=image),patch.object(scene,'_recognize_tile_variants',return_value=('7p',.52)),patch.object(scene,'_face_ink_on_paper',return_value=.1):
            result=scene._refine_meld_tile({'name':None,'score':.3,'quad4':[[0,0],[30,0],[30,50],[0,50]]},image,[0,0,30,50],None)
        self.assertEqual(result['name'],'7p')

    def test_pixel_cache_does_not_reuse_a_different_tile(self):
        import numpy as np
        from tile_recognizer import recognize,clear_cache
        clear_cache()
        reader=MagicMock();reader.recognize_tile.return_value=('1m',.9)
        image=np.zeros((10,10,3),dtype='uint8')
        recognize(reader,image);recognize(reader,image.copy())
        self.assertEqual(reader.recognize_tile.call_count,1)
        image[0,0,0]=1;recognize(reader,image)
        self.assertEqual(reader.recognize_tile.call_count,2)

    def test_meld_matching_skips_expensive_optional_hand_quality_work(self):
        import numpy as np
        from tile_recognizer import recognize,clear_cache
        clear_cache();reader=MagicMock()
        def match(image,quality=None):
            if quality is not None:quality['margin']=.2
            return '1m',.9
        reader.recognize_tile.side_effect=match
        image=np.zeros((10,10,3),dtype='uint8')
        recognize(reader,image)
        self.assertNotIn('quality',reader.recognize_tile.call_args.kwargs)
        details={};recognize(reader,image,quality=details)
        self.assertEqual(details['margin'],.2)

    def test_open_hand_shanten_and_waits_account_for_known_meld_tiles(self):
        from open_efficiency import analyze_open
        hand=['1m','2m','3m','4m','5m','6m','7m','8m','9m','1p']
        groups=[[{'name':'7z'}]*3]
        result=analyze_open(hand,groups)
        self.assertEqual(result['shanten'],0)
        self.assertEqual(result['waits'],[{'tile':'1p','remaining':3}])
        drawn=analyze_open(hand+['2z'],groups)
        self.assertEqual(drawn['candidates'][0]['shanten'],0)
        self.assertIn(drawn['candidates'][0]['tile'],['1p','2z'])

    def test_kan_counts_physical_four_but_effective_three_in_efficiency(self):
        from open_efficiency import analyze_open
        hand=['1m','2m','3m','4m','5m','6m','7m','8m','9m','1p']
        result=analyze_open(hand,[[{'name':'7z'}]*4])
        self.assertEqual(result['shanten'],0)
        self.assertEqual(result['waits'],[{'tile':'1p','remaining':3}])


class StrategyTests(unittest.TestCase):
    def test_whole_table_context_early_advice_and_environment_are_preserved(self):
        local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]},'warnings':[],
               'rivers':{'across':[{'name':'4m','score':.9}]},
               'melds':{'across':{'groups':[[{'name':'7z','score':.9}]*3]}},'dora_indicators':['4p']}
        answer={'discard':'发','reason':'孤张退让保留手牌','stance':'均衡','offense':'门清推进争取立直',
                'defense':'对家碰中，四万是对家现物；其他家不保证安全','next':'摸到对子可保留，若有人立直先比现物',
                'alternatives':[{'tile':'白','when':'摸到发对子','reason':'保留对子'}],
                'call_plan':'有役且能明显加速时再鸣牌','placement':'东一局优先牌效率',
                'threats':['对家碰中，有役推进但未确认听牌'],
                'table':{'round':'东一局','self_wind':'南','remaining':42,'scores':{'own':25000},'riichi':{'across':None}}}
        client=SparkClient(Settings(api_key='test-only'))
        events=[];payloads=[]
        def request(payload,callback,parse_mode):
            payloads.append(payload)
            callback({'phase':'writing','partial_text':'{"discard":"发","reason":"孤张退让保留手牌",'})
            self.assertTrue(any(e.get('phase')=='advice' for e in events))
            return {'report':answer,'elapsed_seconds':1,'usage':{},'text':json.dumps(answer)}
        with patch.object(client,'_request_with_retry',side_effect=request):
            with Image.new('RGB',(1600,900),'#302020') as image:
                result=analyze_table(client,image,local,history=[{'round':'东一局'}],progress=events.append)
        serialized=json.dumps(payloads[0],ensure_ascii=False)
        self.assertIn('四万',serialized);self.assertIn('中',serialized)
        self.assertIn('safety_by_player',serialized);self.assertIn('recent_context',serialized)
        self.assertEqual(result['report']['observed']['remaining'],42)
        self.assertEqual(result['report']['strategy']['recommendation']['tile'],'6z')
        self.assertTrue(result['report']['strategy']['table_threats'])
        self.assertTrue(result['report']['strategy']['alternatives'])
        self.assertEqual(payloads[0]['reasoning_effort'],'low')
        self.assertTrue(payloads[0]['chat_template_kwargs']['enable_thinking'])

    def test_selfcheck_write_denial_returns_status_without_crash_dialog(self):
        import desktop_app
        fake=MagicMock();fake.analyze.return_value={'total':14}
        with patch.object(sys,'argv',['desktop_app.py','--self-check','--check-output','not-writable.json']),patch.object(desktop_app,'ExactSolver',return_value=fake),patch.object(desktop_app,'load_settings',return_value=Settings()),patch.object(Path,'write_text',side_effect=PermissionError()),patch.object(desktop_app,'log_event'),patch.object(desktop_app.messagebox,'showerror') as popup:
            self.assertEqual(desktop_app.main(),2)
            popup.assert_not_called()

    def test_failed_atomic_save_keeps_previous_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'settings.json';initial=Settings(api_key='test-only')
            save_settings(initial,path);before=path.read_bytes()
            with patch('desktop_runtime.os.replace',side_effect=PermissionError()):
                with self.assertRaises(PermissionError):save_settings(Settings(model='new',api_key='new-test-only'),path)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(load_settings(path).model,initial.model)

    def test_unwritable_default_file_uses_encrypted_current_user_storage(self):
        import desktop_runtime as runtime
        import winreg
        key=r'Software\MahjongStrategyAssistant\Tests'+chr(92)+uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as folder,patch.object(runtime,'REGISTRY_SETTINGS_KEY',key),patch.dict('os.environ',{'LOCALAPPDATA':folder}):
            settings=Settings(api_key='test-only-registry-key',model='saved-fallback')
            try:
                with patch.object(runtime.os,'replace',side_effect=PermissionError()):save_settings(settings)
                self.assertEqual(load_settings().model,'saved-fallback')
                self.assertEqual(load_settings().api_key,settings.api_key)
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,key) as handle:
                    raw=winreg.QueryValueEx(handle,'EncryptedSettings')[0]
                    self.assertNotIn(settings.api_key,raw)
            finally:
                try:winreg.DeleteKey(winreg.HKEY_CURRENT_USER,key)
                except FileNotFoundError:pass

    def test_temp_creation_denial_falls_back_without_permission_retry_loop(self):
        import desktop_runtime as runtime
        settings=Settings(api_key='test-only')
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'LOCALAPPDATA':folder}),patch.object(Path,'open',side_effect=PermissionError()) as opened,patch.object(runtime,'_registry_settings',return_value='current-user-storage') as fallback:
            self.assertEqual(save_settings(settings),'current-user-storage')
            self.assertEqual(opened.call_count,1)
            self.assertEqual(fallback.call_count,1)
            self.assertNotIn('api_key',fallback.call_args.args[0])

    def test_portable_encrypted_settings_survive_file_and_registry_denials(self):
        import desktop_runtime as runtime
        original=runtime._write_settings_file
        with tempfile.TemporaryDirectory() as folder:
            portable=Path(folder)/'app'/'user-data'/'settings.json'
            def write(value,target):
                if target!=portable:raise PermissionError()
                return original(value,target)
            def registry(value=None):
                if value is not None:raise PermissionError()
                return None
            with patch.dict('os.environ',{'LOCALAPPDATA':folder}),patch.object(runtime,'portable_settings_path',return_value=portable),patch.object(runtime,'_write_settings_file',side_effect=write),patch.object(runtime,'_registry_settings',side_effect=registry):
                settings=Settings(api_key='test-only-portable',mini_mode=True)
                self.assertEqual(save_settings(settings),str(portable))
                self.assertEqual(load_settings(),settings)
                self.assertNotIn(settings.api_key,portable.read_text(encoding='utf-8'))

    def test_stream_object_parser_respects_escaped_braces(self):
        from spark_strategy import complete_object
        value={'discard':'一万','reason':'保留"{好形}"','stance':'均衡'}
        self.assertEqual(complete_object(json.dumps(value)),value)
        self.assertIsNone(complete_object('{"reason":"not yet closed'))

    def test_impossible_self_draw_pon_wording_is_corrected_and_disclosed(self):
        from strategy_context import repair_call_timing
        strategy={'next_turn_plan':['摸中成三张役牌可碰'],'call_plan':'门清听牌再考虑立直'}
        warnings=repair_call_timing(strategy,['7z','7z'])
        self.assertTrue(warnings)
        self.assertIn('暗刻',strategy['next_turn_plan'][0])
        self.assertIn('他家',strategy['next_turn_plan'][0])
        self.assertNotIn('摸中成三张役牌可碰',strategy['next_turn_plan'][0])

    def test_riichi_and_dragon_meld_rule_contradictions_are_repaired(self):
        from strategy_context import repair_rule_claims
        strategy={'next_turn_plan':['摸五万进一向听可立直'],'call_plan':'碰白失役牌且破门清',
                  'recommendation':{'tile':'4m','reason':'保留两面'}}
        corrections=repair_rule_claims(strategy,HAND)
        self.assertEqual(strategy['recommendation']['tile'],'4m')
        self.assertNotIn('一向听可立直',strategy['next_turn_plan'][0])
        self.assertIn('门清听牌',strategy['next_turn_plan'][0])
        self.assertIn('获得役牌',strategy['call_plan'])
        self.assertEqual(len(corrections),2)
        correct={'call_plan':'一向听不能立直；碰白不会失去役牌'}
        self.assertEqual(repair_rule_claims(correct,HAND),[])
        self.assertEqual(correct['call_plan'],'一向听不能立直；碰白不会失去役牌')

    def test_special_and_regular_routes_remain_distinct_after_discard(self):
        from open_efficiency import closed_route_facts
        hand=['1m','1m','2m','2m','4m','7m','8m','1p','3p','4p','4p','6p','8p','7m']
        result=closed_route_facts(hand,[])
        self.assertEqual(result['current']['普通型'],3)
        self.assertEqual(result['current']['七对子'],2)
        chosen=next(v for v in result['after_discard'] if v['tile']=='4m')
        self.assertEqual(chosen['forms']['普通型'],3)
        self.assertEqual(chosen['forms']['七对子'],2)
        self.assertNotIn('5p',result['seven_pairs']['new_pair_draws'])
        self.assertIn('3p',result['seven_pairs']['new_pair_draws'])

    def test_placement_arithmetic_does_not_inherit_a_wrong_model_rank(self):
        from strategy_context import check_table_claims
        strategy={'placement':'本家24500居第三，追对家下家62000，宜速攻','defense':'对家未立直'}
        notes=check_table_claims(strategy,{'scores':{'own':24500,'right':62000,'across':62000,'left':51500},'riichi':{}})
        self.assertIn('暂列第4',strategy['placement'])
        self.assertIn('27000',strategy['placement'])
        self.assertNotIn('第三',strategy['placement'])
        self.assertIn('宜速攻',strategy['placement'])
        self.assertEqual(strategy['defense'],'对家立直状态未确认')
        self.assertTrue(notes)

    def test_number_tile_pon_does_not_inherit_dragon_yakuhai(self):
        from strategy_context import repair_rule_claims
        strategy={'call_plan':'碰中/六万得役牌但破门清，视速度再定'}
        notes=repair_rule_claims(strategy,['7z','7z','6m','6m'])
        self.assertIn('碰中可得役牌',strategy['call_plan'])
        self.assertIn('碰六万本身不成役',strategy['call_plan'])
        self.assertTrue(notes)


if __name__=='__main__':unittest.main(verbosity=2)
