"""Desktop contracts; all model network traffic here goes to an in-process fake."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,HTTPServer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from desktop_runtime import (Settings,ExactSolver,save_settings,load_settings,dora_from_indicator,
                             build_seen,log_event,acquire_single_instance)
from spark_strategy import SparkClient,parse_strategy,parse_json_object,validate_report
from window_capture import WindowCaptureError
from desktop_app import compact_calculations,report_sections,audit_numeric_claims
from table_watcher import signature,difference,TableWatcher
from PIL import Image


HAND=['1m','2m','3m','4m','5m','6m','7m','8m','9m','2p','3p','4p','5p','4z']
def report(hand=None,tile='4z'):
    return {'observed':{'hand':hand or HAND,'round':'东1局','self_wind':'西','remaining':69,'scores':{},'dora_indicators':[], 'melds':{}},
            'strategy':{'stance':'均衡','recommendation':{'action':'打牌','tile':tile,'reason':'保留进张'},'offense':'门清推进','defense':'观察牌河',
                        'alternatives':[],'next_turn_plan':['听牌后看局势'],'call_plan':'暂不鸣牌','placement':'按点差调整','uncertainties':[],'confidence':'medium'}}


class DesktopTests(unittest.TestCase):
    def test_thinking_cannot_be_disabled(self):
        with self.assertRaises(ValueError):Settings(reasoning_effort='none').validate()
        with self.assertRaises(ValueError):Settings(reasoning_effort='extreme').validate()
        self.assertEqual(Settings().reasoning_effort,'medium')

    def test_encrypted_settings_round_trip_without_plaintext_key(self):
        if os.name!='nt':self.skipTest('Windows DPAPI')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json';settings=Settings(api_key='unit-test-private-key')
            save_settings(settings,path)
            self.assertNotIn(settings.api_key,path.read_text(encoding='utf-8'))
            self.assertEqual(load_settings(path).api_key,settings.api_key)

    def test_solver_runs_without_node_and_preserves_known_result(self):
        solver=ExactSolver()
        try:
            value=solver.analyze(HAND)
            self.assertEqual(value['discards'][0]['tile'],30)
            self.assertEqual(compact_calculations(value)['candidates'][0]['tile'],'4z')
        finally:solver.close()

    def test_structured_strategy_parser_rejects_incomplete_data(self):
        good=report()
        self.assertEqual(parse_strategy('```json\n'+json.dumps(good)+'\n```'),good)
        for value in ('not JSON','{}','{"observed":{},"strategy":{}}'):
            with self.subTest(value=value),self.assertRaises(ValueError):parse_strategy(value)

    def test_json_syntax_repair_is_explicit_and_does_not_execute_text(self):
        parsed,repaired=parse_json_object('{"strategy": {"stance": "均衡",}}')
        self.assertTrue(repaired);self.assertEqual(parsed['strategy']['stance'],'均衡')

    def test_indicator_conversion_uses_single_honor_cycle(self):
        self.assertEqual([dora_from_indicator(x) for x in ['4p','4z','2s']],['5p','5z','3s'])
        # 北 wraps into 白 and 中 wraps into 東: one cycle, not two.
        self.assertEqual(dora_from_indicator('7z'),'1z')
        self.assertEqual(dora_from_indicator('4z'),'5z')

    def test_misread_hand_and_invalid_discard_are_flagged(self):
        local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]}}
        self.assertEqual(validate_report(report(),local),[])
        self.assertTrue(validate_report(report(HAND[:-1]+['5z']),local))
        self.assertTrue(validate_report(report(tile='9s'),local))

    def test_thirteen_tiles_cannot_be_presented_as_current_discard(self):
        local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND[:-1]]}}
        self.assertTrue(validate_report(report(HAND[:-1],tile='1m'),local))

    def test_chi_middle_is_inferred_from_two_named_neighbours(self):
        from scene_recognize import _infer_chi_middle
        group=[{'name':'7p','score':.58},{'name':None,'score':.35},{'name':'9p','score':.52}]
        _infer_chi_middle(group)
        self.assertEqual(group[1]['name'],'8p')
        self.assertTrue(group[1]['inferred'])
        self.assertEqual(group[0]['group_confidence'],'low')
        # Gap of one is ambiguous without display direction: stays unnamed.
        group=[{'name':'7p','score':.58},{'name':None,'score':.35},{'name':'8p','score':.52}]
        _infer_chi_middle(group)
        self.assertIsNone(group[1]['name'])
        # Honors and mixed suits never infer.
        group=[{'name':'5z','score':.9},{'name':None,'score':.3},{'name':'7z','score':.5}]
        _infer_chi_middle(group)
        self.assertIsNone(group[1]['name'])

    def test_refine_meld_tile_rejects_low_score_white_dragon_noise(self):
        from scene_recognize import _refine_meld_tile
        import numpy as np

        class FakeReader:
            def __init__(self,name,score):self.name,self.score=name,score
            def recognize_tile(self,image):return self.name,self.score

        image=np.full((60,80,3),210,dtype='uint8')
        image[10:50,20:60]=235
        tile={'name':'5z','score':.42,'quad4':[[20,10],[60,10],[60,50],[20,50]]}
        refined=_refine_meld_tile(tile,image,[0,0,80,60],FakeReader('5z',.40))
        # The blank plate reads 5z weakly; below .85 the name must be dropped
        # instead of polluting the seen-tile counts.
        self.assertIsNone(refined['name'])

    def test_report_has_real_offense_defense_and_plan_sections(self):
        title,sections=report_sections({'report':report(),'validation_warnings':[]})
        self.assertEqual(title,'建议打 北')
        headings=[h for h,b in sections]
        for name in ['进攻路线','防守与风险','后续几巡','顺位与点差']:self.assertIn(name,headings)

    def test_model_cannot_confuse_total_copies_with_distinct_draw_types(self):
        value=report();value['strategy']['recommendation']={'tile':'4m','reason':'进张15种不变'}
        exact={'candidates':[{'tile':'4m','kinds':5,'ukeire':15}]}
        self.assertTrue(audit_numeric_claims(value,exact))
        value['strategy']['recommendation']['reason']='进张5种，共15枚'
        self.assertEqual(audit_numeric_claims(value,exact),[])

    def test_seen_excludes_hand_and_low_confidence_and_caps_copies(self):
        evidence={'rivers':{'right':[{'name':'1m','score':.9},{'name':'1m','score':.2}]},
                  'melds':{'across':{'groups':[[{'name':'1m','score':.8}]]}},
                  'dora_indicators':['2m']}
        codes,audit=build_seen(['1m','1m','5z'],evidence)
        # hand already holds two 1m, so at most two more may be seen.
        self.assertEqual(codes.count('1m'),2)
        self.assertEqual(codes.count('2m'),1)
        self.assertEqual(audit['excluded_low_confidence'],1)
        self.assertEqual(audit['capped_by_physical_limit'],0)
        solver=ExactSolver()
        try:
            valid=['1m','2m','3m','4m','5m','6m','7m','8m','9m','1p','2p','3p','4p']
            value=solver.analyze(valid,codes)
            self.assertIn('discards',value)
        finally:solver.close()

    def test_physical_tile_overflow_and_indicator_confusion_are_flagged(self):
        local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]}}
        value=report();value['observed']['rivers']={'right':['4z']*4}  # hand already holds one 4z
        self.assertTrue(validate_report(value,local))
        clean=report();clean['observed']['dora_indicators']=['4p']
        clean['observed']['dora_tiles']=['5p'];clean['strategy']['recommendation']['reason']='宝牌5p价值高'
        self.assertEqual(validate_report(clean,local),[])
        clean['observed']['dora_tiles']=['4p']
        self.assertTrue(validate_report(clean,local))

    def test_single_instance_mutex_is_exclusive_within_process(self):
        if os.name!='nt':self.skipTest('Windows named mutex')
        name='Local\\MahjongStrategyAssistant_TestMutex'
        self.assertTrue(acquire_single_instance(name))
        self.assertFalse(acquire_single_instance(name))

    def test_diagnostics_log_redacts_credentials(self):
        path=Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData'/'Local'))/'MahjongStrategyAssistant'/'diagnostics.log'
        before=path.read_text(encoding='utf-8') if path.exists() else ''
        log_event('request failed: Bearer abc123secret api_key=xyz789')
        text=path.read_text(encoding='utf-8')[len(before):]
        self.assertNotIn('abc123secret',text)
        self.assertNotIn('xyz789',text)

    def test_stage_timings_are_reported_without_disabling_thinking(self):
        received=[]
        answer=json.dumps(report(),ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                payload={'choices':[{'delta':{'content':answer},'finish_reason':None}]}
                self.wfile.write(('data: '+json.dumps(payload)+'\n\n').encode());self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]}}
            with Image.new('RGB',(640,360),'navy') as image:
                result=SparkClient(settings).analyze(image,local)
            self.assertIn('vision',result['stage_seconds'])
            self.assertIn('strategy',result['stage_seconds'])
            self.assertIn('vision',result['stage_usage'])
            self.assertTrue(all(request['chat_template_kwargs']['enable_thinking'] for request in received))
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_fast_mode_uses_one_small_image_low_thinking_and_maps_action(self):
        received=[]
        fast_answer=json.dumps({'stance':'均衡','discard':'4z','reason':'效率最高','offense':'推进','defense':'观察','next':'听牌','dora':'无'},ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                payload={'choices':[{'delta':{'content':fast_answer},'finish_reason':None}]}
                self.wfile.write(('data: '+json.dumps(payload)+'\n\n').encode());self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]},'dora_indicators':[],
                   'rivers':{},'melds':{},'warnings':[]}
            with Image.new('RGB',(2560,1400),'navy') as image:
                result=SparkClient(settings).analyze_fast(image,local)
            self.assertEqual(len(received),1)  # single call, not two
            request=received[0]
            self.assertEqual(request['reasoning_effort'],'low')
            self.assertTrue(request['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(request['max_tokens'],280)
            images=[p for p in request['messages'][1]['content'] if p['type']=='image_url']
            self.assertEqual(len(images),1)
            self.assertLessEqual(len(images[0]['image_url']['url']),250000)  # compressed
            report=result['report']
            self.assertEqual(report['strategy']['recommendation']['tile'],'4z')
            self.assertEqual(report['observed']['hand'],HAND)
            self.assertEqual(result['mode'],'fast')
            self.assertIn('fast',result['stage_seconds'])
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_fast_mode_thirteen_tiles_forces_wait(self):
        received=[]
        answer=json.dumps({'stance':'进攻','discard':'1m','reason':'x','offense':'y','defense':'z','next':'w','dora':'d'},ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':answer}}]})+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND[:-1]]},'dora_indicators':[],'rivers':{},'melds':{},'warnings':[]}
            with Image.new('RGB',(640,360),'navy') as image:
                result=SparkClient(settings).analyze_fast(image,local)
            action=result['report']['strategy']['recommendation']
            self.assertEqual(action['action'],'等待摸牌')
            self.assertIsNone(action['tile'])
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_analysis_mode_setting_defaults_to_structured_and_validates(self):
        self.assertEqual(Settings().analysis_mode,'structured')
        with self.assertRaises(ValueError):Settings(analysis_mode='turbo').validate()

    def test_dora_panel_fallback_reads_fixture_indicators(self):
        folder=os.environ.get('MJ_USER_FIXTURES')
        if not folder:self.skipTest('Set MJ_USER_FIXTURES to the private sample directory')
        from scene_recognize import read_scene
        with Image.open(Path(folder)/'用户样例-1.png') as game:
            game.info['game_bbox']=[0,0,game.width,game.height]
            result=read_scene(game)
        indicators=result['dora_indicators']
        # The skin's leaf-motif empty slots defeat the upstream colored-slot
        # reader; the white-face panel fallback must recover the indicators.
        self.assertTrue(indicators,f'panel fallback returned no indicators: {result["warnings"]}')
        self.assertLessEqual(len(indicators),5)
        from desktop_runtime import tile_index
        for name in indicators:
            tile_index(name)  # must be valid tile codes

    def test_structured_state_carries_local_riichi_detection(self):
        received=[]
        fast_answer=json.dumps({'stance':'防守','discard':None,'reason':'双立直威胁','offense':'后攻','defense':'现物','next':'安全','dora':'无'},ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':fast_answer},'finish_reason':None}]})+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]},'dora_indicators':[],
                   'rivers':{},'melds':{},'warnings':[],'riichi':{'across':True}}
            with Image.new('RGB',(640,360),'navy') as image:
                result=SparkClient(settings).analyze_structured(image,local)
            user_text=json.dumps(received[0]['messages'][1]['content'],ensure_ascii=False)
            self.assertIn('riichi',user_text)
            self.assertIn('本地检测到立直棒',user_text)
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_transient_stream_failure_retries_once_and_recovers(self):
        received=[]
        state={'broken':1}
        fast_answer=json.dumps({'stance':'进攻','discard':'1m','reason':'进张最多','offense':'推进','defense':'现物','next':'听牌','dora':'无'},ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                if state['broken']>0:
                    # A dropped mid-stream failure: one malformed chunk, no DONE.
                    state['broken']-=1
                    self.wfile.write(b'data: {not-json\n\n');self.wfile.flush()
                    self.wfile.close()
                    return
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':fast_answer},'finish_reason':None}]})+'\n\n').encode())
                self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]},'dora_indicators':[],
                   'rivers':{},'melds':{},'warnings':[]}
            with Image.new('RGB',(640,360),'navy') as image:
                result=SparkClient(settings).analyze_fast(image,local)
            self.assertEqual(len(received),2)  # one retry after the broken stream
            self.assertEqual(result['report']['strategy']['recommendation']['tile'],'1m')
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_occlusion_error_has_machine_readable_code(self):
        error=WindowCaptureError('window_occluded','遮挡')
        self.assertEqual(error.code,'window_occluded')
        self.assertIn('遮挡',str(error))

    def test_table_signature_detects_change_and_ignores_identical_frames(self):
        with Image.new('RGB',(1280,720),'#203020') as a, Image.new('RGB',(1280,720),'#203020') as b:
            self.assertLess(difference(signature(a),signature(b)),0.001)
        with Image.new('RGB',(1280,720),'#c02020') as changed:
            self.assertGreater(difference(signature(a),signature(changed)),0.05)

    def test_table_watcher_fires_only_after_stable_change_and_respects_cooldown(self):
        frames=[Image.new('RGB',(640,360),'#202020'),Image.new('RGB',(640,360),'#202020'),
                Image.new('RGB',(640,360),'#c02020'),Image.new('RGB',(640,360),'#c02020'),
                Image.new('RGB',(640,360),'#c02020')]
        calls=[]
        def grab():
            if not frames:
                watcher_ref[0].stop()
                return None
            return frames.pop(0)
        watcher_ref=[None]
        watcher=TableWatcher(grab,lambda:calls.append(1),interval=0.001,threshold=0.01,cooldown=999)
        watcher_ref[0]=watcher
        watcher._loop()  # deterministic single pass through available frames
        self.assertEqual(len(calls),1)

    def test_structured_mode_defaults_and_rejects_invalid(self):
        self.assertEqual(Settings().analysis_mode,'structured')
        with self.assertRaises(ValueError):Settings(analysis_mode='network').validate()

    def test_structured_mode_sends_no_image_and_uses_program_facts(self):
        received=[]
        answer=json.dumps({'stance':'进攻','discard':'4z','reason':'效率高','offense':'推进','defense':'观察','next':'听牌','dora':'无'},ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':answer}}]})+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]},'dora_indicators':['4p'],
                   'rivers':{},'melds':{},'warnings':['测试警告']}
            calc={'shanten':0,'candidates':[{'tile':'4z','shanten':0,'ukeire':8}]}
            with Image.new('RGB',(2560,1400),'navy') as image:
                result=SparkClient(settings).analyze_structured(image,local,calc)
            self.assertEqual(len(received),1)
            request=received[0]
            self.assertEqual(request['reasoning_effort'],'low')
            self.assertTrue(request['chat_template_kwargs']['enable_thinking'])
            user_text=json.dumps(request['messages'][1]['content'],ensure_ascii=False)
            # Hybrid mode: one small image for environment context, with the
            # verified facts JSON still in the same message.
            self.assertIn('image_url',user_text)
            self.assertIn('hand_total',user_text)  # program facts JSON
            self.assertIn('5p',user_text)      # program dora conversion
            self.assertIn('shanten',user_text) # program calculation
            self.assertIn('测试警告',user_text)
            self.assertEqual(result['mode'],'structured')
            self.assertEqual(result['report']['observed']['dora_tiles'],['5p'])
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_spark_request_is_vision_streaming_and_thinking_enabled(self):
        received=[]
        answer=json.dumps(report(),ensure_ascii=False)
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for delta in [{'reasoning_content':'test internal reasoning'},{'content':answer}]:
                    payload={'choices':[{'delta':delta,'finish_reason':None}]}
                    self.wfile.write(('data: '+json.dumps(payload)+'\n\n').encode());self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            local={'hand':{'reliable':True,'tiles':[{'name':t} for t in HAND]}}
            with Image.new('RGB',(640,360),'navy') as image:result=SparkClient(settings).analyze(image,local)
            self.assertTrue(result['reasoning_used'])
            self.assertNotIn('test internal reasoning',json.dumps(result))
            self.assertEqual(len(received),2)
            request=received[0]
            self.assertEqual(request['reasoning_effort'],'low')
            self.assertEqual(received[1]['reasoning_effort'],'medium')
            self.assertTrue(request['chat_template_kwargs']['enable_thinking'])
            self.assertTrue(request['stream'])
            self.assertGreaterEqual(sum(part['type']=='image_url' for part in request['messages'][1]['content']),4)
        finally:server.shutdown();server.server_close();thread.join(timeout=2)


class UserScreenshotTests(unittest.TestCase):
    def test_attached_screenshots_have_correct_local_hands(self):
        folder=os.environ.get('MJ_USER_FIXTURES')
        if not folder:self.skipTest('Set MJ_USER_FIXTURES to the private sample directory')
        from scene_recognize import read_concealed
        expected=[['3m','6m','8m','6p','7p','8p','1s','0s','9s','1z','2z','5z','7z'],
                  ['1m','1m','2m','2m','4m','7m','8m','1p','3p','4p','4p','6p','8p','7m']]
        for i,hand in enumerate(expected,1):
            with Image.open(Path(folder)/('用户样例-%d.png'%i)) as image:result=read_concealed(image)
            self.assertEqual([t['name'] for t in result['tiles']],hand)
            self.assertTrue(result['reliable'])

    def test_full_screen_offsets_use_window_relative_roi(self):
        folder=os.environ.get('MJ_USER_FIXTURES')
        if not folder:self.skipTest('Set MJ_USER_FIXTURES to the private sample directory')
        from scene_recognize import read_scene
        with Image.open(Path(folder)/'用户样例-2.png') as game:
            desktop=Image.new('RGB',(3200,1800),'#202020')
            desktop.paste(game,(200,100))
            desktop.info['game_bbox']=[200,100,200+game.width,100+game.height]
        result=read_scene(desktop)
        self.assertEqual(result['hand']['count'],14)
        self.assertTrue(result['hand']['reliable'])
        self.assertEqual(result['hand']['tiles'][0]['name'],'1m')
        self.assertGreaterEqual(result['hand']['tiles'][0]['box'][0],552)

    def test_open_hand_with_meld_is_reliable_and_waits(self):
        # A called hand: 10 concealed tiles + one 3-tile meld = 13 total.
        local={'hand':{'reliable':False,'count':10,'warnings':['手牌为10张'],
                       'tiles':[{'name':t} for t in ['2p','5p','6p','9p','6s','7s','9s','5z','6z','6z']]},
               'melds':{'own':{'groups':[[{'name':'3z'},{'name':'3z'},{'name':'3z'}]]}},
               'rivers':{},'dora_indicators':[]}
        from scene_recognize import read_scene
        image=Image.new('RGB',(1920,1080),'#203040')
        upgraded=Image.new('RGB',(1920,1080),'#203040')
        # Simulate the read_scene upgrade path on evidence shaped like its output.
        own_meld_tiles=sum(len(g) for g in local['melds']['own']['groups'])
        total=local['hand']['count']+own_meld_tiles
        self.assertEqual(total,13)
        local['hand']['reliable']=True
        local['hand']['open_hand']={'concealed':10,'meld_tiles':own_meld_tiles,'total':total}
        answer=json.dumps({'stance':'均衡','discard':'6p','reason':'孤张','offense':'推进','defense':'观察','next':'听牌','dora':'无'},ensure_ascii=False)
        received=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':answer}}]})+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            settings=Settings(base_url='http://127.0.0.1:%d/v1'%server.server_port,api_key='test-key')
            with Image.new('RGB',(1588,844),'navy') as image:
                result=SparkClient(settings).analyze_structured(image,local)
            self.assertEqual(result['report']['strategy']['recommendation']['action'],'等待摸牌')
            self.assertIsNone(result['report']['strategy']['recommendation']['tile'])
            user_text=json.dumps(received[0]['messages'][1]['content'],ensure_ascii=False)
            self.assertIn('hand_total',user_text)
            self.assertIn('13',user_text)
        finally:server.shutdown();server.server_close();thread.join(timeout=2)

    def test_validate_report_accepts_open_hand_discard(self):
        open_hand=['2p','5p','6p','9p','6s','7s','9s','5z','6z','6z']
        local={'hand':{'reliable':True,'tiles':[{'name':t} for t in open_hand]}}
        value=report(hand=open_hand,tile='2p')
        value['observed']['melds']={'own':[{'type':'pon','tiles':['3z','3z','3z']}]}
        self.assertEqual(validate_report(value,local),[])

    def test_custom_window_resolutions_are_normalized(self):
        folder=os.environ.get('MJ_USER_FIXTURES')
        if not folder:self.skipTest('Set MJ_USER_FIXTURES to the private sample directory')
        from scene_recognize import read_concealed
        expected=['1m','1m','2m','2m','4m','7m','8m','1p','3p','4p','4p','6p','8p','7m']
        with Image.open(Path(folder)/'用户样例-2.png') as source:
            base=source.convert('RGB')
        w,h=base.size
        for width in (1280,1588,1920,2560,3840):
            height=round(h*width/w)
            image=base.resize((width,height),Image.Resampling.LANCZOS) if width!=w else base
            result=read_concealed(image)
            with self.subTest(width=width):
                self.assertEqual([t['name'] for t in result['tiles']],expected)
                self.assertTrue(result['reliable'])
                self.assertTrue(all(0<=t['box'][0]<width and 0<=t['box'][1]<height for t in result['tiles']))

    def test_custom_resolution_full_screen_bbox(self):
        folder=os.environ.get('MJ_USER_FIXTURES')
        if not folder:self.skipTest('Set MJ_USER_FIXTURES to the private sample directory')
        from scene_recognize import read_scene
        expected=['1m','1m','2m','2m','4m','7m','8m','1p','3p','4p','4p','6p','8p','7m']
        with Image.open(Path(folder)/'用户样例-2.png') as source:
            game=source.convert('RGB').resize((1588,844),Image.Resampling.LANCZOS)
        desktop=Image.new('RGB',(2560,1440),'#303030')
        desktop.paste(game,(600,300))
        desktop.info['game_bbox']=[600,300,600+1588,300+844]
        result=read_scene(desktop)
        self.assertEqual([t['name'] for t in result['hand']['tiles']],expected)
        self.assertTrue(result['hand']['reliable'])
        self.assertEqual(result['viewport'],[600,300,2188,1144])


if __name__=='__main__':unittest.main(verbosity=2)
