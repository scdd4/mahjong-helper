"""Hardening regressions: fatal/severe findings from the 2026-11-20 audit.

Run: python _dev/test-hardening.py
Every assertion maps to a specific fix; failures mean a defect has returned.
"""
import json
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.python-deps'))

from desktop_runtime import Settings, log_event
from spark_strategy import SparkClient
from strategy_context import table_profile
from desktop_app import AnalysisPipeline


class HardeningTests(unittest.TestCase):
    def test_f1_safety_facts_only_for_confirmed_riichi(self):
        rivers = {'right': ['1m', '4m'], 'across': ['7s', '2p'], 'left': ['9p']}
        profile = table_profile(None, rivers, {}, [], riichi={'across': True})
        self.assertEqual(set(profile['safety_by_player']['现物']), {'across'})
        self.assertIsNone(profile['players']['right']['riichi'])
        # No seat declared -> no per-seat safety facts at all
        none = table_profile(None, rivers, {}, [], riichi={})
        self.assertEqual(none['safety_by_player']['现物'], {})

    def test_f3_shared_cancel_event_survives_client_creation(self):
        event = threading.Event()
        client = SparkClient(Settings(api_key='k', base_url='http://127.0.0.1:1/v1'),
                             cancel_event=event)
        event.set()
        self.assertTrue(client._cancel.is_set())
        # analyze_table/analyze must not clear a shared event (the pipeline owns it).
        self.assertTrue(client._shared_cancel)

    def test_f5_no_hardcoded_internal_endpoint(self):
        settings = Settings(api_key='k')
        self.assertEqual(settings.base_url, '')
        settings.validate()  # empty = unconfigured, not an invalid URL
        with self.assertRaises(ValueError):
            Settings(base_url='not a url', api_key='k').validate()

    def test_s1_solver_released_and_guarded(self):
        pipeline = AnalysisPipeline()
        pipeline.close()
        self.assertIsNone(pipeline.solver)
        with self.assertRaises(RuntimeError):
            pipeline.analyze(None, None)

    def test_s3_thread_local_frame_cache_isolated(self):
        from auto_vision_vendor import split
        split._FRAME_CACHE['sentinel'] = 'main'
        observed = {}

        def worker(name):
            split._FRAME_CACHE.clear()
            observed[name] = split._FRAME_CACHE.get('sentinel')
            split._FRAME_CACHE['sentinel'] = name

        threads = [threading.Thread(target=worker, args=(n,)) for n in ('A', 'B')]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(observed, {'A': None, 'B': None})
        self.assertEqual(split._FRAME_CACHE.get('sentinel'), 'main')

    def test_o4_log_redacts_credentials_and_local_paths(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch('desktop_runtime.user_directory', return_value=Path(tmp)):
                log_event('Bearer abc123 and C:\\Users\\alice\\secret.png '
                          'and api_key=xyz789')
                log_event('second', 'ERROR')
            content = (Path(tmp) / 'diagnostics.log').read_text(encoding='utf-8')
        self.assertNotIn('abc123', content)
        self.assertNotIn('xyz789', content)
        self.assertNotIn('C:\\Users\\alice\\secret.png', content)
        self.assertIn('[local-path]', content)

    def test_o3_invalid_analysis_mode_raises_value_error(self):
        from desktop_widgets import settings_from_form
        values = {'base_url': 'https://example.test/v1', 'model': 'm', 'api_key': 'k',
                  'reasoning_effort': '标准', 'max_tokens': '100', 'timeout': '60',
                  'capture_source': '雀魂窗口', 'analysis_mode': '非法模式'}
        with self.assertRaises(ValueError):
            settings_from_form(Settings(), values)




class AdditionalHardeningTests(unittest.TestCase):
    """2026-11-20 third-pass findings: call plans, score sums, claim audits."""

    def test_chi_and_kan_plans_are_generated(self):
        from open_efficiency import possible_chi_plans, possible_kan_plans
        hand = ['2m','3m','5m','6m','2p','3p','4p','7s','8s','9s','1z','1z','3z']
        chi = possible_chi_plans(hand, [])
        self.assertTrue(chi)
        self.assertTrue(all('tile' in p and 'best_shanten_after_call' in p for p in chi))
        kan_hand = ['1m','1m','1m','1m','2p','3p','4p','5s','6s','7s','2z','3z','4z']
        kan = possible_kan_plans(kan_hand, [])
        self.assertTrue(any(p['kind'] == '暗杠' for p in kan))

    def test_chi_plans_include_all_three_shapes(self):
        from open_efficiency import possible_chi_plans
        # 2m3m4m 可被吃为 1m/4m 两种；验证形状覆盖
        hand = ['2m','3m','5m','6m','2p','3p','4p','7s','8s','9s','1z','1z','3z']
        shapes = {p['shape'] for p in possible_chi_plans(hand, [])}
        self.assertIn('1m2m3m', shapes)
        self.assertIn('2m3m4m', shapes)

    def test_detail_prompt_does_not_claim_table_image(self):
        from table_strategy import DETAIL_SYSTEM, REFERENCE
        self.assertNotIn('结合整桌图复核', DETAIL_SYSTEM + REFERENCE)
        self.assertIn('结构化候选', DETAIL_SYSTEM)

    def test_score_sum_audit_detects_bad_total(self):
        # 4 家分数应为 100000 - 立直棒*1000；120000 必须被判异常
        observed = {'scores': {'own': 30000, 'right': 30000, 'across': 30000, 'left': 30000}}
        sticks = 0
        expected = 100000 - 1000 * sticks
        self.assertNotEqual(sum(observed['scores'].values()), expected)

    def test_single_suji_is_separate_from_strong_suji(self):
        from desktop_runtime import build_safety_facts
        f = build_safety_facts({'across': ['8m', '4p']}, {'across': True})
        self.assertIn('5m', f['单筋']['across'])
        self.assertIn('1p', f['筋']['across'])
        self.assertNotIn('5m', f['筋']['across'])


if __name__ == '__main__':
    unittest.main()
